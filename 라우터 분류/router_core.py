# SPDX-FileCopyrightText: Copyright 2026 OSSP Router contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize


MODEL_NAMES = ("ax31-light", "ax31", "axk1-think")
TOKEN_RATES = {
    "ax31-light": {"input": 1.0, "output": 4.0},
    "ax31": {"input": 2.127, "output": 8.509},
    "axk1-think": {"input": 6.565, "output": 26.260},
}
TIER_FACTORS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
ARTIFACT_SCHEMA_VERSION = "1.0"


def read_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path | str, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
        os.chmod(output_path, 0o644)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSON object가 아닙니다.")
            records.append(value)
    return records


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value if item is not None and str(item)]
    return [str(value)]


def labels_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    raw = record.get("label")
    if raw is None:
        raw = record.get("labels")
    if raw is None:
        raw = record.get("prediction")
    if raw is None:
        # 초기 데이터의 오타도 읽는다.
        raw = record.get("lable")
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("prediction/labels는 JSON object여야 합니다.")

    languages = _as_list(raw.get("response_language"))
    return {
        "response_language": languages[0] if languages else "unknown",
        "request_types": sorted(set(_as_list(raw.get("request_types")))),
        "content_domains": sorted(set(_as_list(raw.get("content_domains")))),
        # 문자열과 배열 입력을 같은 형태로 맞춘다.
        "answer_formats": sorted(set(_as_list(raw.get("answer_formats")))),
    }


def taxonomy_label_sets(taxonomy: Mapping[str, Any]) -> dict[str, set[str]]:
    def ids(section: str) -> set[str]:
        labels = taxonomy.get(section, {}).get("labels", [])
        return {str(item["id"]) for item in labels}

    return {
        "response_language": ids("language"),
        "request_types": ids("request"),
        "content_domains": ids("content"),
        "answer_formats": ids("answer_format"),
    }


def validate_labels(
    records: Iterable[Mapping[str, Any]],
    allowed: Mapping[str, set[str]],
    *,
    strict: bool = True,
) -> list[str]:
    errors: list[str] = []
    for index, record in enumerate(records):
        labels = labels_from_record(record)
        episode_id = record.get("episode_id", index)
        language = labels["response_language"]
        if language != "unknown" and language not in allowed["response_language"]:
            errors.append(f"{episode_id}: unknown response_language={language!r}")
        for key in ("request_types", "content_domains", "answer_formats"):
            unknown = sorted(set(labels[key]) - allowed[key])
            if unknown:
                errors.append(f"{episode_id}: unknown {key}={unknown!r}")
    if strict and errors:
        preview = "\n".join(errors[:20])
        suffix = "" if len(errors) <= 20 else f"\n... 외 {len(errors) - 20}건"
        raise ValueError(f"taxonomy에 없는 라벨이 있습니다.\n{preview}{suffix}")
    return errors


def require_runtime_labels(records: Iterable[Mapping[str, Any]]) -> None:
    for index, record in enumerate(records):
        raw = next(
            (
                record.get(key)
                for key in ("label", "labels", "prediction", "lable")
                if record.get(key) is not None
            ),
            None,
        )
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"record {index}: label object가 필요합니다. "
                "실행 입력은 episode_id + prompt + label입니다."
            )


# 이전 모델과의 호환 이름
require_runtime_predictions = require_runtime_labels


_WORD_RE = re.compile(r"\w+", re.UNICODE)
_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
CHAR_FEATURE_LIMIT = 4_000


def _char_feature_view(prompt: str) -> str:
    if len(prompt) <= CHAR_FEATURE_LIMIT:
        return prompt
    # 앞 지시와 끝 질문을 남긴다.
    return prompt[:3_000] + "\n[...TRUNCATED_FOR_CHAR_FEATURES...]\n" + prompt[-1_000:]


def _numeric_features(prompt: str) -> list[float]:
    char_count = len(prompt)
    safe_length = max(char_count, 1)
    utf8_bytes = len(prompt.encode("utf-8"))
    word_count = len(_WORD_RE.findall(prompt))
    whitespace = sum(char.isspace() for char in prompt)
    digits = sum(char.isdigit() for char in prompt)
    ascii_letters = sum(char.isascii() and char.isalpha() for char in prompt)
    hangul = len(_HANGUL_RE.findall(prompt))
    punctuation = sum(not char.isalnum() and not char.isspace() for char in prompt)
    lines = prompt.count("\n") + 1
    marker_view = _char_feature_view(prompt)
    return [
        float(char_count),
        float(utf8_bytes),
        float(word_count),
        float(whitespace),
        float(digits),
        float(ascii_letters),
        float(hangul),
        float(punctuation),
        float(lines),
        math.log1p(char_count),
        math.log1p(utf8_bytes),
        math.log1p(word_count),
        digits / safe_length,
        ascii_letters / safe_length,
        hangul / safe_length,
        punctuation / safe_length,
        float("```" in marker_view),
        float("def " in marker_view or "function " in marker_view),
        float("Question:" in marker_view or "질문:" in marker_view),
        float(any(marker in marker_view for marker in ("A.", "A)", "①"))),
    ]


def _category_features(record: Mapping[str, Any]) -> dict[str, float]:
    labels = labels_from_record(record)
    features: dict[str, float] = {
        f"language={labels['response_language']}": 1.0,
    }
    for value in labels["request_types"]:
        features[f"request={value}"] = 1.0
    for value in labels["content_domains"]:
        features[f"content={value}"] = 1.0
    for value in labels["answer_formats"]:
        features[f"format={value}"] = 1.0

    # hard label과 분류 확률을 함께 쓴다.
    probability_axes = record.get("classification_probabilities")
    if isinstance(probability_axes, Mapping):
        for axis, mapping in probability_axes.items():
            if not isinstance(axis, str) or not isinstance(mapping, Mapping):
                continue
            finite_values: list[float] = []
            for label, raw_value in mapping.items():
                if not isinstance(label, str):
                    continue
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(value):
                    continue
                value = float(np.clip(value, 0.0, 1.0))
                features[f"prob:{axis}={label}"] = value
                finite_values.append(value)
            if finite_values:
                values = np.asarray(finite_values, dtype=np.float64)
                features[f"prob-summary:{axis}:max"] = float(values.max())
                features[f"prob-summary:{axis}:margin"] = float(
                    values.max()
                    - (np.partition(values, -2)[-2] if len(values) > 1 else 0.0)
                )
                clipped = np.clip(values, 1e-9, 1.0 - 1e-9)
                features[f"prob-summary:{axis}:entropy"] = float(
                    -np.mean(
                        clipped * np.log(clipped)
                        + (1.0 - clipped) * np.log(1.0 - clipped)
                    )
                )
    return features


class FeatureBuilder:

    def __init__(self) -> None:
        self.word_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents=None,
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            min_df=2,
            max_features=25_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.char_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=30_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.category_vectorizer = DictVectorizer(sparse=True, dtype=np.float32)
        self.numeric_scaler = StandardScaler()

    @staticmethod
    def _prompts(records: Sequence[Mapping[str, Any]]) -> list[str]:
        prompts: list[str] = []
        for index, record in enumerate(records):
            prompt = record.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError(f"record {index}: prompt가 문자열이 아닙니다.")
            prompts.append(prompt)
        return prompts

    def fit(self, records: Sequence[Mapping[str, Any]]) -> "FeatureBuilder":
        prompts = self._prompts(records)
        self.word_vectorizer.fit(prompts)
        self.char_vectorizer.fit([_char_feature_view(prompt) for prompt in prompts])
        self.category_vectorizer.fit([_category_features(record) for record in records])
        numeric = np.asarray([_numeric_features(prompt) for prompt in prompts], dtype=np.float64)
        self.numeric_scaler.fit(numeric)
        return self

    def transform(self, records: Sequence[Mapping[str, Any]]) -> csr_matrix:
        prompts = self._prompts(records)
        word = self.word_vectorizer.transform(prompts)
        char = self.char_vectorizer.transform(
            [_char_feature_view(prompt) for prompt in prompts]
        )
        category = self.category_vectorizer.transform(
            [_category_features(record) for record in records]
        )
        numeric_raw = np.asarray(
            [_numeric_features(prompt) for prompt in prompts], dtype=np.float64
        )
        numeric = csr_matrix(self.numeric_scaler.transform(numeric_raw), dtype=np.float32)
        return hstack((word, char, category, numeric), format="csr", dtype=np.float32)

    def fit_transform(self, records: Sequence[Mapping[str, Any]]) -> csr_matrix:
        return self.fit(records).transform(records)


class DenseOutputTokenRegressor:

    def __init__(
        self,
        *,
        text_feature_count: int,
        svd: Any,
        estimator: Any,
        calibration_factor: float,
    ) -> None:
        self.text_feature_count = int(text_feature_count)
        self.svd = svd
        self.estimator = estimator
        self.calibration_factor = float(calibration_factor)

    def _dense_features(self, features: csr_matrix) -> np.ndarray:
        text = normalize(features[:, : self.text_feature_count], norm="l2")
        semantic = self.svd.transform(text)
        structural = features[:, self.text_feature_count :].toarray()
        return np.column_stack((structural, semantic)).astype(np.float32, copy=False)

    def predict_tokens(self, features: csr_matrix) -> np.ndarray:
        prediction = self.estimator.predict(self._dense_features(features))
        return np.clip(prediction * self.calibration_factor, 1.0, None)


class BlendedStructuralInputTokenRegressor:

    def __init__(
        self,
        *,
        tokenizer_json: str,
        linear_estimator: Any,
        tree_estimator: Any,
        linear_calibration_factor: float,
        tree_calibration_factor: float,
        tree_weight: float,
        final_calibration_factor: float,
    ) -> None:
        self.tokenizer_json = tokenizer_json
        self.linear_estimator = linear_estimator
        self.tree_estimator = tree_estimator
        self.linear_calibration_factor = float(linear_calibration_factor)
        self.tree_calibration_factor = float(tree_calibration_factor)
        self.tree_weight = float(tree_weight)
        self.final_calibration_factor = float(final_calibration_factor)
        self._tokenizer = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_tokenizer"] = None
        return state

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            try:
                from tokenizers import Tokenizer
            except ImportError as error:
                raise RuntimeError(
                    "정확한 input-token 예측에는 tokenizers 패키지가 필요합니다. "
                    "requirements.txt를 설치하세요."
                ) from error
            self._tokenizer = Tokenizer.from_str(self.tokenizer_json)
        return self._tokenizer

    def _features(self, records: Sequence[Mapping[str, Any]]) -> np.ndarray:
        prompts = [str(record["prompt"]) for record in records]
        encodings = self._get_tokenizer().encode_batch(prompts, add_special_tokens=False)
        token_count = np.asarray([len(encoding.ids) for encoding in encodings], dtype=np.float64)
        numeric = np.asarray([_numeric_features(prompt) for prompt in prompts], dtype=np.float64)
        hinges = np.column_stack(
            [np.maximum(token_count - cut, 0.0) for cut in (64, 128, 256, 512, 1024, 2048)]
        )
        return np.column_stack(
            (numeric, token_count, np.log1p(token_count), np.sqrt(token_count), hinges)
        )

    def predict_tokens(
        self, features: csr_matrix, records: Sequence[Mapping[str, Any]]
    ) -> np.ndarray:
        del features
        dense = self._features(records)
        linear = np.clip(
            self.linear_estimator.predict(dense) * self.linear_calibration_factor,
            1.0,
            None,
        )
        tree = np.clip(
            self.tree_estimator.predict(dense) * self.tree_calibration_factor,
            1.0,
            None,
        )
        blended = linear * (1.0 - self.tree_weight) + tree * self.tree_weight
        return np.clip(blended * self.final_calibration_factor, 1.0, None)


class LinearStructuralInputTokenRegressor:

    def __init__(
        self,
        *,
        tokenizer_json: str,
        estimator: Any,
        calibration_factor: float,
    ) -> None:
        self.tokenizer_json = tokenizer_json
        self.estimator = estimator
        self.calibration_factor = float(calibration_factor)
        self._tokenizer = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_tokenizer"] = None
        return state

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            try:
                from tokenizers import Tokenizer
            except ImportError as error:
                raise RuntimeError("input-token 예측에는 tokenizers가 필요합니다.") from error
            self._tokenizer = Tokenizer.from_str(self.tokenizer_json)
        return self._tokenizer

    def build_dense_features(
        self, records: Sequence[Mapping[str, Any]]
    ) -> np.ndarray:
        prompts = [str(record["prompt"]) for record in records]
        encodings = self._get_tokenizer().encode_batch(
            prompts, add_special_tokens=False
        )
        token_count = np.asarray(
            [len(encoding.ids) for encoding in encodings], dtype=np.float64
        )
        numeric = np.asarray(
            [_numeric_features(prompt) for prompt in prompts], dtype=np.float64
        )
        hinges = np.column_stack(
            [
                np.maximum(token_count - cut, 0.0)
                for cut in (64, 128, 256, 512, 1024, 2048)
            ]
        )
        return np.column_stack(
            (numeric, token_count, np.log1p(token_count), np.sqrt(token_count), hinges)
        )

    def predict_from_dense(self, dense: np.ndarray) -> np.ndarray:
        prediction = self.estimator.predict(dense)
        return np.clip(prediction * self.calibration_factor, 1.0, None)

    def predict_tokens(
        self, features: csr_matrix, records: Sequence[Mapping[str, Any]]
    ) -> np.ndarray:
        del features
        return self.predict_from_dense(self.build_dense_features(records))


class BlendedRidgeKNNOutputTokenRegressor:

    def __init__(
        self,
        *,
        text_feature_count: int,
        ridge_estimator: Any,
        ridge_transform: str,
        ridge_calibration_factor: float,
        reference_text: csr_matrix,
        reference_log_targets: np.ndarray,
        neighbors: int,
        knn_calibration_factor: float,
        knn_weight: float,
        final_calibration_factor: float,
    ) -> None:
        self.text_feature_count = int(text_feature_count)
        self.ridge_estimator = ridge_estimator
        self.ridge_transform = ridge_transform
        self.ridge_calibration_factor = float(ridge_calibration_factor)
        self.reference_text = reference_text
        self.reference_log_targets = np.asarray(reference_log_targets, dtype=np.float64)
        self.neighbors = int(neighbors)
        self.knn_calibration_factor = float(knn_calibration_factor)
        self.knn_weight = float(knn_weight)
        self.final_calibration_factor = float(final_calibration_factor)

    def _knn_prediction(self, features: csr_matrix) -> np.ndarray:
        query = normalize(features[:, : self.text_feature_count], norm="l2")
        similarities = (query @ self.reference_text.T).toarray()
        count = min(self.neighbors, self.reference_text.shape[0])
        positions = np.argpartition(similarities, -count, axis=1)[:, -count:]
        result = np.zeros(features.shape[0], dtype=np.float64)
        for row in range(features.shape[0]):
            weights = np.maximum(similarities[row, positions[row]], 1e-3) ** 2
            result[row] = np.expm1(
                np.average(self.reference_log_targets[positions[row]], weights=weights)
            )
        return np.clip(result * self.knn_calibration_factor, 1.0, None)

    def predict_tokens(self, features: csr_matrix) -> np.ndarray:
        ridge_raw = self.ridge_estimator.predict(features)
        ridge = inverse_target_prediction(ridge_raw, self.ridge_transform)
        ridge = np.clip(ridge * self.ridge_calibration_factor, 1.0, None)
        knn = self._knn_prediction(features)
        blended = ridge * (1.0 - self.knn_weight) + knn * self.knn_weight
        return np.clip(blended * self.final_calibration_factor, 1.0, None)


class TierQualityGainModel:

    def __init__(
        self,
        *,
        text_feature_count: int,
        ridge_estimators: Mapping[str, Any],
        reference_text: csr_matrix,
        reference_targets: Mapping[str, np.ndarray],
        tier_configs: Mapping[str, Mapping[str, float | int]],
    ) -> None:
        self.text_feature_count = int(text_feature_count)
        self.ridge_estimators = dict(ridge_estimators)
        self.reference_text = reference_text
        self.reference_targets = {
            key: np.asarray(value, dtype=np.float64)
            for key, value in reference_targets.items()
        }
        self.tier_configs = {
            key: dict(value) for key, value in tier_configs.items()
        }

    def predict(self, features: csr_matrix, tier: str) -> np.ndarray:
        if tier not in self.tier_configs:
            raise ValueError(f"지원하지 않는 quality gain tier: {tier}")
        config = self.tier_configs[tier]
        tier_estimators = self.ridge_estimators.get(tier)
        estimators = (
            tier_estimators
            if isinstance(tier_estimators, Mapping)
            else self.ridge_estimators
        )
        neighbors = int(config["neighbors"])
        power = float(config["similarity_power"])
        weight = float(config["knn_weight"])
        similarity_floor = float(config.get("similarity_floor", 0.0))
        uncertainty_penalty = float(config.get("uncertainty_penalty", 0.0))
        if neighbors <= 0 or power <= 0.0 or not 0.0 <= weight <= 1.0:
            raise ValueError(f"{tier} quality gain 설정이 올바르지 않습니다.")
        if not 0.0 <= similarity_floor < 1.0 or uncertainty_penalty < 0.0:
            raise ValueError(f"{tier} quality risk 설정이 올바르지 않습니다.")

        query = normalize(features[:, : self.text_feature_count], norm="l2")
        similarities = (query @ self.reference_text.T).toarray()
        count = min(neighbors, self.reference_text.shape[0])
        positions = np.argpartition(similarities, -count, axis=1)[:, -count:]
        neighbor_similarities = np.take_along_axis(
            similarities, positions, axis=1
        )
        weights = np.maximum(neighbor_similarities, 1e-4) ** power
        weight_totals = np.maximum(weights.sum(axis=1), 1e-12)
        maximum_similarity = np.max(neighbor_similarities, axis=1)
        similarity_reliability = np.clip(
            (maximum_similarity - similarity_floor)
            / max(1.0 - similarity_floor, 1e-12),
            0.0,
            1.0,
        )
        effective_knn_weight = weight * similarity_reliability
        result: dict[str, np.ndarray] = {}
        for target_name in ("ax_minus_light", "think_minus_light"):
            ridge = np.asarray(
                estimators[target_name].predict(features),
                dtype=np.float64,
            )
            reference_target = self.reference_targets[target_name]
            # 두 품질 target은 같은 이웃과 가중치를 쓴다.
            knn = (
                (reference_target[positions] * weights).sum(axis=1)
                / weight_totals
            )
            prediction = (
                ridge * (1.0 - effective_knn_weight)
                + knn * effective_knn_weight
            )
            if uncertainty_penalty > 0.0:
                neighbor_values = reference_target[positions]
                neighbor_mean = (
                    (neighbor_values * weights).sum(axis=1) / weight_totals
                )
                neighbor_variance = (
                    ((neighbor_values - neighbor_mean[:, None]) ** 2 * weights).sum(axis=1)
                    / weight_totals
                )
                estimator = estimators[target_name]
                if hasattr(estimator, "predict_uncertainty"):
                    model_uncertainty = np.asarray(
                        estimator.predict_uncertainty(features), dtype=np.float64
                    )
                else:
                    model_uncertainty = np.zeros(features.shape[0], dtype=np.float64)
                combined_uncertainty = np.sqrt(
                    np.maximum(neighbor_variance, 0.0) * effective_knn_weight
                    + model_uncertainty**2 * (1.0 - effective_knn_weight)
                )
                prediction = prediction - uncertainty_penalty * combined_uncertainty
            result[target_name] = prediction
        return np.column_stack(
            (
                np.zeros(features.shape[0], dtype=np.float64),
                result["ax_minus_light"],
                result["think_minus_light"],
            )
        )


class HurdleGainEstimator:

    def __init__(
        self,
        *,
        positive_classifier: Any,
        negative_classifier: Any,
        positive_regressor: Any,
        negative_regressor: Any,
        positive_default: float,
        negative_default: float,
        residual_scale: float,
    ) -> None:
        self.positive_classifier = positive_classifier
        self.negative_classifier = negative_classifier
        self.positive_regressor = positive_regressor
        self.negative_regressor = negative_regressor
        self.positive_default = float(positive_default)
        self.negative_default = float(negative_default)
        self.residual_scale = float(max(residual_scale, 0.0))

    @staticmethod
    def _positive_probability(classifier: Any, features: csr_matrix) -> np.ndarray:
        probabilities = np.asarray(classifier.predict_proba(features), dtype=np.float64)
        classes = list(classifier.classes_)
        return probabilities[:, classes.index(1)]

    def _components(
        self, features: csr_matrix
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        positive_probability = self._positive_probability(
            self.positive_classifier, features
        )
        negative_probability = self._positive_probability(
            self.negative_classifier, features
        )
        probability_total = positive_probability + negative_probability
        overflow = probability_total > 0.98
        if np.any(overflow):
            scale = 0.98 / probability_total[overflow]
            positive_probability[overflow] *= scale
            negative_probability[overflow] *= scale
        positive = (
            np.asarray(self.positive_regressor.predict(features), dtype=np.float64)
            if self.positive_regressor is not None
            else np.full(features.shape[0], self.positive_default, dtype=np.float64)
        )
        negative = (
            np.asarray(self.negative_regressor.predict(features), dtype=np.float64)
            if self.negative_regressor is not None
            else np.full(features.shape[0], self.negative_default, dtype=np.float64)
        )
        return (
            np.clip(positive_probability, 0.0, 1.0),
            np.clip(negative_probability, 0.0, 1.0),
            np.clip(positive, 0.0, 1.0),
            np.clip(negative, 0.0, 1.0),
        )

    def predict(self, features: csr_matrix) -> np.ndarray:
        pos_prob, neg_prob, pos_size, neg_size = self._components(features)
        return pos_prob * pos_size - neg_prob * neg_size

    def predict_uncertainty(self, features: csr_matrix) -> np.ndarray:
        pos_prob, neg_prob, pos_size, neg_size = self._components(features)
        mean = pos_prob * pos_size - neg_prob * neg_size
        second_moment = pos_prob * pos_size**2 + neg_prob * neg_size**2
        variance = np.maximum(second_moment - mean**2, 0.0)
        return np.sqrt(variance + self.residual_scale**2)


class BatchCostRiskModel:

    def __init__(
        self,
        *,
        residual_ratios: np.ndarray,
        increment_residual_ratios: np.ndarray | None = None,
        calibration_bands: Mapping[str, np.ndarray],
        quantile: float = 0.95,
        draws: int = 256,
        seed: int = 20260826,
    ) -> None:
        values = np.asarray(residual_ratios, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(MODEL_NAMES):
            raise ValueError("batch cost residual_ratios 형태가 올바르지 않습니다.")
        self.residual_ratios = np.clip(values, 0.05, 20.0)
        if increment_residual_ratios is None:
            self.increment_residual_ratios = self.residual_ratios.copy()
            self.increment_mode = False
        else:
            increments = np.asarray(increment_residual_ratios, dtype=np.float64)
            if increments.shape != values.shape:
                raise ValueError("batch cost increment residual 형태가 올바르지 않습니다.")
            self.increment_residual_ratios = np.clip(increments, 0.05, 20.0)
            self.increment_mode = True
        self.calibration_bands = {
            model: np.asarray(calibration_bands[model], dtype=np.int8)
            for model in MODEL_NAMES
        }
        if any(len(value) != len(values) for value in self.calibration_bands.values()):
            raise ValueError("batch cost calibration band 길이가 올바르지 않습니다.")
        self.quantile = float(quantile)
        self.draws = int(draws)
        self.seed = int(seed)
        if not 0.5 < self.quantile < 1.0 or self.draws < 32:
            raise ValueError("batch cost quantile/draws 설정이 올바르지 않습니다.")

    def estimate_ratio(
        self,
        selected: np.ndarray,
        raw_costs: np.ndarray,
        runtime_bands: Mapping[str, np.ndarray],
    ) -> tuple[float, float]:
        selected = np.asarray(selected, dtype=np.int64)
        raw_costs = np.asarray(raw_costs, dtype=np.float64)
        if raw_costs.shape != (len(selected), len(MODEL_NAMES)):
            raise ValueError("batch cost runtime cost 형태가 올바르지 않습니다.")
        light_runtime_bands = np.asarray(
            runtime_bands[MODEL_NAMES[0]], dtype=np.int64
        )
        rng = np.random.default_rng(self.seed)
        denominator = np.zeros(self.draws, dtype=np.float64)
        calibration_light_bands = self.calibration_bands[MODEL_NAMES[0]]

        if self.increment_mode:
            # light 비용과 승급분의 오차를 따로 재표본화한다.
            for light_band in range(4):
                current = np.flatnonzero(light_runtime_bands == light_band)
                if len(current) == 0:
                    continue
                pool = np.flatnonzero(calibration_light_bands == light_band)
                if len(pool) < 12:
                    pool = np.arange(len(self.residual_ratios), dtype=np.int64)
                sampled = pool[
                    rng.integers(0, len(pool), size=(self.draws, len(current)))
                ]
                denominator += (
                    self.residual_ratios[sampled, 0]
                    * raw_costs[current, 0][None, :]
                ).sum(axis=1)

            extra_total = np.zeros(self.draws, dtype=np.float64)
            for model_index, model_name in enumerate(MODEL_NAMES[1:], start=1):
                runtime_model_bands = np.asarray(
                    runtime_bands[model_name], dtype=np.int64
                )
                calibration_model_bands = self.calibration_bands[model_name]
                model_rows = np.flatnonzero(selected == model_index)
                for model_band in range(4):
                    for light_band in range(4):
                        current = model_rows[
                            (runtime_model_bands[model_rows] == model_band)
                            & (light_runtime_bands[model_rows] == light_band)
                        ]
                        if len(current) == 0:
                            continue
                        pool = np.flatnonzero(
                            (calibration_model_bands == model_band)
                            & (calibration_light_bands == light_band)
                        )
                        if len(pool) < 12:
                            pool = np.flatnonzero(
                                calibration_model_bands == model_band
                            )
                        if len(pool) < 12:
                            pool = np.arange(
                                len(self.residual_ratios), dtype=np.int64
                            )
                        sampled = pool[
                            rng.integers(
                                0, len(pool), size=(self.draws, len(current))
                            )
                        ]
                        predicted_extra = np.maximum(
                            raw_costs[current, model_index]
                            - raw_costs[current, 0],
                            0.0,
                        )
                        extra_total += (
                            self.increment_residual_ratios[sampled, model_index]
                            * predicted_extra[None, :]
                        ).sum(axis=1)
            ratios = 1.0 + extra_total / np.maximum(denominator, 1e-15)
            return float(np.mean(ratios)), float(
                np.quantile(ratios, self.quantile)
            )

        numerator = np.zeros(self.draws, dtype=np.float64)

        for model_index, model_name in enumerate(MODEL_NAMES):
            runtime_model_bands = np.asarray(runtime_bands[model_name], dtype=np.int64)
            calibration_model_bands = self.calibration_bands[model_name]
            model_rows = np.flatnonzero(selected == model_index)
            if len(model_rows) == 0:
                continue
            for model_band in range(4):
                for light_band in range(4):
                    current = model_rows[
                        (runtime_model_bands[model_rows] == model_band)
                        & (light_runtime_bands[model_rows] == light_band)
                    ]
                    if len(current) == 0:
                        continue
                    pool = np.flatnonzero(
                        (calibration_model_bands == model_band)
                        & (calibration_light_bands == light_band)
                    )
                    if len(pool) < 12:
                        pool = np.flatnonzero(calibration_model_bands == model_band)
                    if len(pool) < 12:
                        pool = np.arange(len(self.residual_ratios), dtype=np.int64)
                    sampled = pool[
                        rng.integers(0, len(pool), size=(self.draws, len(current)))
                    ]
                    numerator += (
                        self.residual_ratios[sampled, model_index]
                        * raw_costs[current, model_index][None, :]
                    ).sum(axis=1)
                    denominator += (
                        self.residual_ratios[sampled, 0]
                        * raw_costs[current, 0][None, :]
                    ).sum(axis=1)
        ratios = numerator / np.maximum(denominator, 1e-15)
        return float(np.mean(ratios)), float(np.quantile(ratios, self.quantile))


def inverse_target_prediction(values: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if transform == "log1p":
        # exp overflow 방지
        return np.expm1(np.clip(values, -20.0, 20.0))
    if transform == "identity":
        return values
    raise ValueError(f"지원하지 않는 target transform: {transform}")


def predict_artifact_targets(
    artifact: Mapping[str, Any],
    features: csr_matrix,
    records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {name: {} for name in MODEL_NAMES}
    shared_input_dense: np.ndarray | None = None
    if records is not None:
        input_estimators = [
            artifact["target_models"][f"{model_name}.input_tokens"].get(
                "estimator"
            )
            for model_name in MODEL_NAMES
        ]
        if all(
            isinstance(estimator, LinearStructuralInputTokenRegressor)
            for estimator in input_estimators
        ):
            tokenizer_jsons = {
                estimator.tokenizer_json for estimator in input_estimators
            }
            if len(tokenizer_jsons) == 1:
                shared_input_dense = input_estimators[0].build_dense_features(
                    records
                )
    quality_mode = artifact.get("quality_prediction_mode", "absolute")
    if isinstance(artifact.get("quality_gain_policy"), Mapping):
        # v2 모델 호환용 quality head
        quality_mode = "absolute"
    for model_name in MODEL_NAMES:
        for target_name in ("input_tokens", "output_tokens", "quality"):
            if (
                target_name == "quality"
                and quality_mode == "delta_chain"
                and model_name != "ax31-light"
            ):
                continue
            key = f"{model_name}.{target_name}"
            model_info = artifact["target_models"][key]
            if (
                target_name == "input_tokens"
                and model_info.get("prediction_kind") == "direct_tokens"
            ):
                if records is None:
                    raise ValueError("direct input-token 예측에는 원본 records가 필요합니다.")
                estimator = model_info["estimator"]
                if (
                    shared_input_dense is not None
                    and isinstance(estimator, LinearStructuralInputTokenRegressor)
                ):
                    values = estimator.predict_from_dense(shared_input_dense)
                else:
                    values = estimator.predict_tokens(features, records)
            elif (
                target_name == "output_tokens"
                and model_info.get("prediction_kind") == "direct_tokens"
            ):
                values = model_info["estimator"].predict_tokens(features)
            else:
                raw = model_info["estimator"].predict(features)
                values = inverse_target_prediction(raw, model_info["transform"])
                values = values * float(model_info.get("calibration_factor", 1.0))
            if target_name == "quality":
                values = np.clip(values, 0.0, 1.0)
            else:
                values = np.clip(values, 1.0, None)
            result[model_name][target_name] = values

    if quality_mode == "delta_chain":
        def predict_delta(key: str) -> np.ndarray:
            model_info = artifact["target_models"][key]
            raw = model_info["estimator"].predict(features)
            return inverse_target_prediction(raw, model_info["transform"])

        light_quality = result["ax31-light"]["quality"]
        ax31_quality = np.clip(
            light_quality + predict_delta("quality_delta.ax31_minus_light"),
            0.0,
            1.0,
        )
        think_quality = np.clip(
            ax31_quality + predict_delta("quality_delta.think_minus_ax31"),
            0.0,
            1.0,
        )
        result["ax31"]["quality"] = ax31_quality
        result["axk1-think"]["quality"] = think_quality
    return result


def predict_output_length_policy_tokens(
    artifact: Mapping[str, Any], features: csr_matrix
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    """Predict conservative output costs from four discrete length bands.

    A policy stores one classifier per model.  Each predicted band is charged
    at the learned 90th-percentile output-token count for that band.  Returning
    ``None`` keeps old artifacts fully compatible with the standard regression
    cost path.
    """
    policy = artifact.get("output_length_policy")
    if not isinstance(policy, Mapping):
        return None
    if policy.get("version") != 1 or policy.get("risk_quantile") != 0.90:
        raise ValueError("지원하지 않는 output_length_policy입니다.")
    entries = policy.get("models")
    if not isinstance(entries, Mapping):
        raise ValueError("output_length_policy.models가 없습니다.")

    tokens: dict[str, np.ndarray] = {}
    bands: dict[str, np.ndarray] = {}
    for model_name in MODEL_NAMES:
        entry = entries.get(model_name)
        if not isinstance(entry, Mapping):
            raise ValueError(f"output_length_policy에 {model_name} 정책이 없습니다.")
        estimator = entry.get("estimator")
        p90 = np.asarray(entry.get("band_p90_tokens"), dtype=np.float64)
        safety_multiplier = float(entry.get("safety_multiplier", 1.0))
        if (
            estimator is None
            or p90.shape != (4,)
            or not np.all(np.isfinite(p90))
            or not np.isfinite(safety_multiplier)
            or safety_multiplier < 1.0
        ):
            raise ValueError(f"{model_name} output_length_policy가 올바르지 않습니다.")
        predicted_bands = np.asarray(estimator.predict(features), dtype=np.int64)
        if predicted_bands.shape != (features.shape[0],) or np.any((predicted_bands < 0) | (predicted_bands > 3)):
            raise ValueError(f"{model_name} 출력 길이 band 예측이 올바르지 않습니다.")
        bands[model_name] = predicted_bands
        tokens[model_name] = np.clip(p90[predicted_bands] * safety_multiplier, 1.0, None)
    return tokens, bands


def model_cost(model_name: str, input_tokens: float, output_tokens: float) -> float:
    rates = TOKEN_RATES[model_name]
    return (
        float(input_tokens) * rates["input"]
        + float(output_tokens) * rates["output"]
    ) / 1_000_000.0


@dataclass
class OptimizationResult:
    selected_indices: np.ndarray
    status: str
    total_cost: float
    total_quality: float
    budget: float


def _repair_to_budget(
    selected: np.ndarray,
    costs: np.ndarray,
    qualities: np.ndarray,
    budget: float,
) -> np.ndarray:
    selected = selected.astype(int, copy=True)
    tolerance = max(1e-15, abs(budget) * 1e-12)
    while float(costs[np.arange(len(selected)), selected].sum()) > budget + tolerance:
        current_cost = costs[np.arange(len(selected)), selected]
        current_quality = qualities[np.arange(len(selected)), selected]
        best: tuple[float, float, int, int] | None = None
        for row in range(len(selected)):
            for candidate in range(costs.shape[1]):
                saving = current_cost[row] - costs[row, candidate]
                if saving <= tolerance:
                    continue
                loss = current_quality[row] - qualities[row, candidate]
                ratio = max(loss, 0.0) / saving
                choice = (ratio, loss, row, candidate)
                if best is None or choice < best:
                    best = choice
        if best is None:
            raise RuntimeError("예산 이내로 복구할 수 있는 더 저렴한 모델 선택이 없습니다.")
        selected[best[2]] = best[3]
    return selected


def _greedy_fallback(
    costs: np.ndarray, qualities: np.ndarray, budget: float
) -> np.ndarray:
    rows = costs.shape[0]
    selected = np.zeros(rows, dtype=int)
    total = float(costs[:, 0].sum())

    candidates: list[tuple[float, float, int, int]] = []
    for row in range(rows):
        for candidate in range(1, costs.shape[1]):
            extra_cost = float(costs[row, candidate] - costs[row, 0])
            gain = float(qualities[row, candidate] - qualities[row, 0])
            if gain <= 0:
                continue
            if extra_cost <= 0:
                priority = math.inf
            else:
                priority = gain / extra_cost
            candidates.append((priority, gain, row, candidate))

    candidates.sort(reverse=True)
    for _, gain, row, candidate in candidates:
        if selected[row] != 0:
            continue
        extra_cost = float(costs[row, candidate] - costs[row, 0])
        if gain > 0 and total + extra_cost <= budget:
            selected[row] = candidate
            total += extra_cost
    return _repair_to_budget(selected, costs, qualities, budget)


def _refine_feasible_selection(
    selected: np.ndarray,
    costs: np.ndarray,
    qualities: np.ndarray,
    budget: float,
) -> np.ndarray:
    selected = selected.astype(int, copy=True)
    rows = len(selected)
    tolerance = max(1e-15, abs(budget) * 1e-12)
    for _ in range(rows * costs.shape[1] + 1):
        current_costs = costs[np.arange(rows), selected]
        current_qualities = qualities[np.arange(rows), selected]
        total_cost = float(current_costs.sum())
        remaining = budget - total_cost
        best: tuple[float, float, float, int, int] | None = None
        for row in range(rows):
            for candidate in range(costs.shape[1]):
                if candidate == selected[row]:
                    continue
                gain = float(qualities[row, candidate] - current_qualities[row])
                extra = float(costs[row, candidate] - current_costs[row])
                if gain <= 1e-12 or extra > remaining + tolerance:
                    continue
                ratio = math.inf if extra <= 0 else gain / extra
                choice = (ratio, gain, -extra, row, candidate)
                if best is None or choice > best:
                    best = choice
        if best is None:
            break
        selected[best[3]] = best[4]
    return _repair_to_budget(selected, costs, qualities, budget)


def _fast_lagrangian_selection(
    costs: np.ndarray, qualities: np.ndarray, budget: float
) -> np.ndarray:
    rows = costs.shape[0]
    row_indices = np.arange(rows)
    tolerance = max(1e-15, abs(budget) * 1e-12)

    def choose(multiplier: float) -> np.ndarray:
        adjusted = qualities - multiplier * costs
        # 동점이면 낮은 비용을 고른다.
        adjusted -= costs * 1e-12
        return adjusted.argmax(axis=1).astype(int)

    unconstrained = choose(0.0)
    unconstrained_cost = float(costs[row_indices, unconstrained].sum())
    if unconstrained_cost <= budget + tolerance:
        return unconstrained

    low = 0.0
    high = 1.0
    feasible_candidates: list[np.ndarray] = []
    infeasible_candidates: list[np.ndarray] = [unconstrained]
    while high < 1e18:
        selected = choose(high)
        total = float(costs[row_indices, selected].sum())
        if total <= budget + tolerance:
            feasible_candidates.append(selected)
            break
        infeasible_candidates.append(selected)
        high *= 2.0

    if not feasible_candidates:
        return _greedy_fallback(costs, qualities, budget)

    for _ in range(64):
        middle = (low + high) / 2.0
        selected = choose(middle)
        total = float(costs[row_indices, selected].sum())
        if total > budget + tolerance:
            low = middle
            infeasible_candidates.append(selected)
        else:
            high = middle
            feasible_candidates.append(selected)

    # 경계 양쪽의 해만 다시 채운다.
    candidate_solutions: list[np.ndarray] = []
    seen: set[bytes] = set()
    for selected in feasible_candidates[-2:]:
        key = selected.tobytes()
        if key not in seen:
            seen.add(key)
            candidate_solutions.append(selected)
    for selected in infeasible_candidates[-2:]:
        key = selected.tobytes()
        if key not in seen:
            seen.add(key)
            candidate_solutions.append(
                _repair_to_budget(selected, costs, qualities, budget)
            )

    best_selection: np.ndarray | None = None
    best_quality = -math.inf
    best_cost = math.inf
    for selected in candidate_solutions:
        refined = _refine_feasible_selection(selected, costs, qualities, budget)
        total_quality = float(qualities[row_indices, refined].sum())
        total_cost = float(costs[row_indices, refined].sum())
        if total_quality > best_quality + 1e-12 or (
            abs(total_quality - best_quality) <= 1e-12 and total_cost < best_cost
        ):
            best_selection = refined
            best_quality = total_quality
            best_cost = total_cost
    assert best_selection is not None
    return best_selection


def optimize_assignments(
    costs: np.ndarray,
    qualities: np.ndarray,
    budget: float,
    *,
    time_limit_seconds: float = 120.0,
    method: str = "fast",
) -> OptimizationResult:
    costs = np.asarray(costs, dtype=np.float64)
    qualities = np.asarray(qualities, dtype=np.float64)
    if costs.shape != qualities.shape or costs.ndim != 2:
        raise ValueError("costs와 qualities는 같은 2차원 배열이어야 합니다.")
    if costs.shape[1] != len(MODEL_NAMES):
        raise ValueError(f"모델 열은 {len(MODEL_NAMES)}개여야 합니다.")
    if not np.all(np.isfinite(costs)) or not np.all(np.isfinite(qualities)):
        raise ValueError("costs/qualities에 NaN 또는 무한대가 있습니다.")
    if np.any(costs < 0):
        raise ValueError("비용은 음수일 수 없습니다.")

    rows, choices = costs.shape
    all_light_cost = float(costs[:, 0].sum())
    if all_light_cost > budget + max(1e-15, abs(budget) * 1e-12):
        raise ValueError("예산이 all-light 기준 비용보다 작아 기본 feasible 해가 없습니다.")
    if rows == 0:
        return OptimizationResult(np.asarray([], dtype=int), "empty", 0.0, 0.0, budget)

    if method == "fast":
        selected = _fast_lagrangian_selection(costs, qualities, budget)
        total_cost = float(costs[np.arange(rows), selected].sum())
        total_quality = float(qualities[np.arange(rows), selected].sum())
        return OptimizationResult(
            selected, "fast_lagrangian", total_cost, total_quality, budget
        )
    if method != "exact":
        raise ValueError("optimizer method는 'fast' 또는 'exact'여야 합니다.")

    # exact 모드에서만 HiGHS를 불러온다.
    from scipy.optimize import Bounds, LinearConstraint, milp

    # micro-credit 단위로 계산한다.
    scale = 1_000_000.0
    scaled_costs = costs.reshape(-1) * scale
    scaled_budget = budget * scale
    variable_count = rows * choices

    cost_rows = np.zeros(variable_count, dtype=np.int32)
    choice_rows = np.repeat(np.arange(1, rows + 1, dtype=np.int32), choices)
    column_indices = np.arange(variable_count, dtype=np.int32)
    matrix = coo_matrix(
        (
            np.concatenate((scaled_costs, np.ones(variable_count))),
            (
                np.concatenate((cost_rows, choice_rows)),
                np.concatenate((column_indices, column_indices)),
            ),
        ),
        shape=(rows + 1, variable_count),
    ).tocsr()
    lower = np.concatenate(([-np.inf], np.ones(rows)))
    upper = np.concatenate(([scaled_budget], np.ones(rows)))

    # 같은 품질이면 싼 모델을 고른다.
    normalized_cost = scaled_costs / max(scaled_budget, 1.0)
    objective = -qualities.reshape(-1) + normalized_cost * 1e-10
    result = milp(
        c=objective,
        integrality=np.ones(variable_count, dtype=np.int8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(matrix, lower, upper),
        options={
            "presolve": True,
            "time_limit": float(time_limit_seconds),
            "mip_rel_gap": 0.0,
        },
    )

    if result.x is not None:
        selected = np.asarray(result.x).reshape(rows, choices).argmax(axis=1)
        selected = _repair_to_budget(selected, costs, qualities, budget)
        status = "exact_optimal" if result.success else f"milp_status_{result.status}_repaired"
    else:
        selected = _greedy_fallback(costs, qualities, budget)
        status = f"greedy_fallback_after_milp_status_{result.status}"

    total_cost = float(costs[np.arange(rows), selected].sum())
    total_quality = float(qualities[np.arange(rows), selected].sum())
    tolerance = max(1e-15, abs(budget) * 1e-12)
    if total_cost > budget + tolerance:
        raise RuntimeError(
            f"내부 오류: 최종 비용 {total_cost}가 예산 {budget}을 초과했습니다."
        )
    return OptimizationResult(selected, status, total_cost, total_quality, budget)


def parse_input_payload(payload: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata: dict[str, Any] = {}
    records: list[dict[str, Any]] = []

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("JSON 배열의 각 원소는 object여야 합니다.")
            if "prompt" in item:
                records.append(item)
            else:
                metadata.update(item)
    elif isinstance(payload, dict):
        if "prompt" in payload:
            records = [payload]
        else:
            collection_key = next(
                (
                    key
                    for key in ("episodes", "data", "items", "prompts", "records")
                    if isinstance(payload.get(key), list)
                ),
                None,
            )
            if collection_key is None:
                raise ValueError(
                    "프롬프트 배열을 찾지 못했습니다. episodes/data/items/prompts/records 중 하나를 사용하세요."
                )
            records = payload[collection_key]
            metadata = {key: value for key, value in payload.items() if key != collection_key}
    else:
        raise ValueError("최상위 JSON은 object 또는 array여야 합니다.")

    if not records:
        raise ValueError("입력 JSON에 프롬프트가 없습니다.")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"record {index}: JSON object가 아닙니다.")
        if not isinstance(record.get("prompt"), str):
            raise ValueError(f"record {index}: prompt 문자열이 없습니다.")
    return metadata, records
