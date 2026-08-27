# SPDX-FileCopyrightText: Copyright 2026 OSSP Router contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from router_core import (
    ARTIFACT_SCHEMA_VERSION,
    MODEL_NAMES,
    TOKEN_RATES,
    TIER_FACTORS,
    optimize_assignments,
    parse_input_payload,
    predict_artifact_targets,
    predict_output_length_policy_tokens,
    read_json,
    require_runtime_labels,
    validate_labels,
    write_json,
)


DEFAULT_CHALLENGE_ID = "ossp-2026-llm-router-challenge"
DEFAULT_SPLIT = "final"
POLICY_ID = "ossp-2026-prompt-router-v1"
OUTPUT_SCHEMA_VERSION = 1


# Public Dev에서 고른 배포 정책이다.
TIER_RISK_POLICY = {
    "fast": {
        "budget_ratio_cap": 1.23,
        "allowed_models": ("ax31-light", "ax31"),
        "band_alphas": (0.05, 0.10, 0.20, 0.40),
        "think_risk_mode": "all",
        "minimum_quality_gain": {"ax31": 0.08, "axk1-think": 0.09},
        "classification_confidence_penalty": 0.0,
    },
    "balanced": {
        "budget_ratio_cap": 1.80,
        "allowed_models": MODEL_NAMES,
        "band_alphas": (0.00, 0.02, 0.05, 0.15),
        "think_risk_mode": "none",
        "minimum_quality_gain": {"ax31": 0.05, "axk1-think": 0.08},
        "classification_confidence_penalty": 0.0,
    },
    "premium": {
        "budget_ratio_cap": 3.20,
        "allowed_models": MODEL_NAMES,
        "band_alphas": (0.00, 0.05, 0.10, 0.25),
        "think_risk_mode": "none",
        "minimum_quality_gain": {"ax31": 0.05, "axk1-think": 0.06},
        "classification_confidence_penalty": 0.0,
    },
}
def _classification_confidence(records: list[dict[str, Any]]) -> np.ndarray:
    values = np.ones(len(records), dtype=np.float64)
    single_axes = {"response_language", "answer_formats"}
    for row_index, record in enumerate(records):
        axes = record.get("classification_probabilities")
        if not isinstance(axes, dict) or not axes:
            continue
        axis_confidence: list[float] = []
        for axis, mapping in axes.items():
            if not isinstance(mapping, dict) or not mapping:
                continue
            probabilities = np.asarray(list(mapping.values()), dtype=np.float64)
            if axis in single_axes:
                axis_confidence.append(float(np.max(probabilities)))
            else:
                axis_confidence.append(
                    float(np.mean(np.abs(probabilities - 0.5) * 2.0))
                )
        if axis_confidence:
            values[row_index] = float(np.mean(axis_confidence))
    return np.clip(values, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select one model for each prompt")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--tier",
        required=True,
        type=str.lower,
        choices=tuple(TIER_FACTORS),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "router_model.joblib",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--budget-safety",
        type=float,
        default=1.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--optimizer",
        choices=("auto", "fast", "exact"),
        default="auto",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--optimizer-time-limit", type=float, default=120.0, help=argparse.SUPPRESS)
    parser.add_argument("--allow-unknown-labels", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _validate_episode_ids(records: list[dict[str, Any]]) -> list[Any]:
    episode_ids: list[Any] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if "episode_id" not in record or record["episode_id"] is None:
            raise ValueError(f"record {index}: episode_id가 필요하며 null일 수 없습니다.")
        episode_id = record["episode_id"]
        if not isinstance(episode_id, str) or not episode_id or len(episode_id) > 128:
            raise ValueError(f"record {index}: episode_id는 1~128자 문자열이어야 합니다.")
        if episode_id in seen:
            raise ValueError(f"중복 episode_id가 있습니다: {episode_id!r}")
        seen.add(episode_id)
        episode_ids.append(episode_id)
    return episode_ids


def _predict_costs_and_qualities(
    artifact: dict[str, Any], records: list[dict[str, Any]], tier: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    features = artifact["feature_builder"].transform(records)
    raw = predict_artifact_targets(artifact, features, records)
    policy_result = predict_output_length_policy_tokens(artifact, features)
    policy_tokens = policy_result[0] if policy_result is not None else None
    policy_bands = policy_result[1] if policy_result is not None else None
    raw_costs = np.zeros((len(records), len(MODEL_NAMES)), dtype=np.float64)
    upper_costs = np.zeros_like(raw_costs)
    qualities = np.zeros_like(raw_costs)
    for model_index, model_name in enumerate(MODEL_NAMES):
        input_tokens = np.round(np.clip(raw[model_name]["input_tokens"], 1.0, None), 3)
        raw_output_source = np.asarray(
            raw[model_name]["output_tokens"], dtype=np.float64
        )
        if policy_bands is not None:
            model_policy = artifact["output_length_policy"]["models"][model_name]
            calibration = np.asarray(
                model_policy.get("band_raw_calibration", [1.0] * 4),
                dtype=np.float64,
            )
            if calibration.shape != (4,) or not np.all(np.isfinite(calibration)):
                raise ValueError(
                    f"{model_name} band_raw_calibration이 올바르지 않습니다."
                )
            raw_output_source = raw_output_source * calibration[
                np.asarray(policy_bands[model_name], dtype=np.int64)
            ]
        raw_output_tokens = np.round(np.clip(raw_output_source, 1.0, None), 3)
        output_source = (
            policy_tokens[model_name] if policy_tokens is not None else raw[model_name]["output_tokens"]
        )
        output_tokens = np.round(np.clip(output_source, 1.0, None), 3)
        qualities[:, model_index] = np.round(
            np.clip(raw[model_name]["quality"], 0.0, 1.0), 6
        )
        rates = TOKEN_RATES[model_name]
        raw_costs[:, model_index] = (
            input_tokens * rates["input"] + raw_output_tokens * rates["output"]
        ) / 1_000_000.0
        upper_costs[:, model_index] = (
            input_tokens * rates["input"] + output_tokens * rates["output"]
        ) / 1_000_000.0

    gain_policy = artifact.get("quality_gain_policy")
    if isinstance(gain_policy, dict):
        if gain_policy.get("version") != 1 or gain_policy.get("model") is None:
            raise ValueError("quality_gain_policy가 올바르지 않습니다.")
        qualities = np.asarray(
            gain_policy["model"].predict(features, tier), dtype=np.float64
        )
        if qualities.shape != raw_costs.shape or not np.all(np.isfinite(qualities)):
            raise ValueError("quality_gain_policy 예측 형태가 올바르지 않습니다.")

    if policy_bands is None:
        raise ValueError("Tier별 단계 보수 정책에는 output_length_policy가 필요합니다.")
    upper_costs = np.maximum(upper_costs, raw_costs)
    return raw_costs, upper_costs, qualities, policy_bands


def _build_tier_costs(
    raw_costs: np.ndarray,
    upper_costs: np.ndarray,
    policy_bands: dict[str, np.ndarray],
    qualities: np.ndarray,
    tier: str,
    budget: float,
) -> np.ndarray:
    policy = TIER_RISK_POLICY[tier]
    alphas = np.asarray(policy["band_alphas"], dtype=np.float64)
    if alphas.shape != (4,) or np.any((alphas < 0.0) | (alphas > 1.0)):
        raise ValueError(f"{tier} band_alphas 설정이 올바르지 않습니다.")

    costs = raw_costs.copy()
    for model_index, model_name in enumerate(MODEL_NAMES[1:], start=1):
        bands = np.asarray(policy_bands[model_name], dtype=np.int64)
        if bands.shape != (raw_costs.shape[0],) or np.any((bands < 0) | (bands >= 4)):
            raise ValueError(f"{model_name} 출력 길이 band가 올바르지 않습니다.")
        risk_bands = bands.copy()
        if model_name == "axk1-think":
            think_risk_mode = str(policy["think_risk_mode"])
            if think_risk_mode == "all":
                risk_bands = np.minimum(risk_bands + 1, 3)
            elif think_risk_mode == "long_only":
                risk_bands = np.where(
                    risk_bands >= 2,
                    np.minimum(risk_bands + 1, 3),
                    risk_bands,
                )
            elif think_risk_mode != "none":
                raise ValueError(f"{tier} think_risk_mode가 올바르지 않습니다.")
        row_alphas = alphas[risk_bands]
        costs[:, model_index] = raw_costs[:, model_index] + row_alphas * (
            upper_costs[:, model_index] - raw_costs[:, model_index]
        )

    allowed = set(policy["allowed_models"])
    gain_setting = policy["minimum_quality_gain"]
    if isinstance(gain_setting, dict):
        missing = set(MODEL_NAMES[1:]) - set(gain_setting)
        if missing:
            raise ValueError(
                f"{tier} minimum_quality_gain에 모델 기준이 없습니다: {sorted(missing)}"
            )
        minimum_quality_gain = {
            model_name: float(gain_setting[model_name])
            for model_name in MODEL_NAMES[1:]
        }
    else:
        minimum_quality_gain = {
            model_name: float(gain_setting) for model_name in MODEL_NAMES[1:]
        }
    if any(
        not np.isfinite(threshold) or threshold < 0.0
        for threshold in minimum_quality_gain.values()
    ):
        raise ValueError(f"{tier} minimum_quality_gain 설정이 올바르지 않습니다.")
    blocked_cost = float(budget + max(1.0, float(costs.sum())))
    for model_index, model_name in enumerate(MODEL_NAMES):
        blocked = np.zeros(raw_costs.shape[0], dtype=bool)
        if model_name not in allowed:
            blocked[:] = True
        elif model_index > 0:
            # 예상 품질 증가가 기준을 넘는 경우만 승급 후보로 둔다.
            blocked = (
                qualities[:, model_index]
                <= qualities[:, 0] + minimum_quality_gain[model_name] + 1e-12
            )
        costs[blocked, model_index] = blocked_cost
    return costs


def _validate_decisions(episode_ids: list[Any], decisions: list[dict[str, Any]]) -> None:
    if len(decisions) != len(episode_ids):
        raise RuntimeError("내부 오류: decisions 수가 입력 episode 수와 다릅니다.")
    if [item["episode_id"] for item in decisions] != episode_ids:
        raise RuntimeError("내부 오류: decisions의 episode_id 또는 순서가 입력과 다릅니다.")
    if any(item["model_id"] not in MODEL_NAMES for item in decisions):
        raise RuntimeError("내부 오류: 허용되지 않은 model_id가 있습니다.")


def main() -> None:
    args = parse_args()
    if not (0.0 < args.budget_safety <= 1.0):
        raise ValueError("--budget-safety는 0보다 크고 1 이하여야 합니다.")
    if args.optimizer_time_limit <= 0:
        raise ValueError("--optimizer-time-limit은 0보다 커야 합니다.")
    optimizer_method = (
        args.optimizer
        if args.optimizer != "auto"
        else ("exact" if args.tier == "fast" else "fast")
    )
    if not args.input.is_file():
        raise FileNotFoundError(f"입력 JSON이 없습니다: {args.input}")
    if not args.model.is_file():
        raise FileNotFoundError(f"학습 모델이 없습니다: {args.model}")

    artifact = joblib.load(args.model)
    if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"지원하지 않는 artifact schema: {artifact.get('schema_version')!r}")

    metadata, records = parse_input_payload(read_json(args.input))
    challenge_id = metadata.get("challenge_id", DEFAULT_CHALLENGE_ID)
    split = metadata.get("split", DEFAULT_SPLIT)
    if not isinstance(challenge_id, str) or not challenge_id.strip():
        raise ValueError("challenge_id는 비어 있지 않은 문자열이어야 합니다.")
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split은 비어 있지 않은 문자열이어야 합니다.")
    require_runtime_labels(records)
    episode_ids = _validate_episode_ids(records)
    allowed = {key: set(values) for key, values in artifact["allowed_labels"].items()}
    validate_labels(records, allowed, strict=not args.allow_unknown_labels)

    # 전역 최적화 순서는 prompt 내용으로만 정한다.
    routing_order = sorted(
        range(len(records)),
        key=lambda index: (
            hashlib.sha256(records[index]["prompt"].encode("utf-8")).digest(),
            records[index]["prompt"],
        ),
    )
    routing_records = [records[index] for index in routing_order]

    raw_costs, upper_costs, qualities, policy_bands = _predict_costs_and_qualities(
        artifact, routing_records, args.tier
    )
    confidence_penalty = float(
        TIER_RISK_POLICY[args.tier]["classification_confidence_penalty"]
    )
    if confidence_penalty > 0.0:
        confidence = _classification_confidence(routing_records)
        qualities[:, 1:] = np.clip(
            qualities[:, 1:] - confidence_penalty * (1.0 - confidence[:, None]),
            0.0,
            1.0,
        )
    light_baseline_cost = float(raw_costs[:, 0].sum())
    policy_cap = float(TIER_RISK_POLICY[args.tier]["budget_ratio_cap"])
    requested_cap = TIER_FACTORS[args.tier] * args.budget_safety
    effective_cap = min(policy_cap, requested_cap)
    budget = light_baseline_cost * effective_cap
    costs = _build_tier_costs(
        raw_costs, upper_costs, policy_bands, qualities, args.tier, budget
    )
    batch_risk_model = artifact.get("batch_cost_risk_model")
    if batch_risk_model is None:
        optimized = optimize_assignments(
            costs,
            qualities,
            budget,
            time_limit_seconds=args.optimizer_time_limit,
            method=optimizer_method,
        )
    else:
        # 선택된 배치 전체의 비용비 95분위수를 맞춘다.
        official_limit = float(TIER_FACTORS[args.tier] * args.budget_safety)
        low_cap = 1.0
        high_cap = effective_cap
        feasible: tuple[float, Any, np.ndarray, float, float] | None = None
        for _ in range(9):
            candidate_cap = (low_cap + high_cap) / 2.0
            candidate_budget = light_baseline_cost * candidate_cap
            candidate_costs = _build_tier_costs(
                raw_costs,
                upper_costs,
                policy_bands,
                qualities,
                args.tier,
                candidate_budget,
            )
            candidate = optimize_assignments(
                candidate_costs,
                qualities,
                candidate_budget,
                method="fast",
            )
            risk_mean, risk_upper = batch_risk_model.estimate_ratio(
                candidate.selected_indices, raw_costs, policy_bands
            )
            if risk_upper <= official_limit + 1e-12:
                feasible = (
                    candidate_cap,
                    candidate,
                    candidate_costs,
                    risk_mean,
                    risk_upper,
                )
                low_cap = candidate_cap
            else:
                high_cap = candidate_cap
        if feasible is None:
            fallback_budget = light_baseline_cost
            fallback_costs = _build_tier_costs(
                raw_costs,
                upper_costs,
                policy_bands,
                qualities,
                args.tier,
                fallback_budget,
            )
            fallback = optimize_assignments(
                fallback_costs, qualities, fallback_budget, method="fast"
            )
            risk_mean, risk_upper = batch_risk_model.estimate_ratio(
                fallback.selected_indices, raw_costs, policy_bands
            )
            feasible = (
                1.0,
                fallback,
                fallback_costs,
                risk_mean,
                risk_upper,
            )
        selected_cap, search_solution, costs, risk_mean, risk_upper = feasible
        budget = light_baseline_cost * selected_cap
        optimized = optimize_assignments(
            costs,
            qualities,
            budget,
            time_limit_seconds=args.optimizer_time_limit,
            method=optimizer_method,
        )
        final_mean, final_upper = batch_risk_model.estimate_ratio(
            optimized.selected_indices, raw_costs, policy_bands
        )
        if final_upper > official_limit + 1e-12:
            optimized = search_solution
            final_mean, final_upper = risk_mean, risk_upper
    row_indices = np.arange(len(records))
    selected_cost = float(costs[row_indices, optimized.selected_indices].sum())
    tolerance = max(1e-15, abs(budget) * 1e-12)
    if selected_cost > budget + tolerance:
        raise RuntimeError("내부 오류: 최종 선택이 Tier 예산을 초과했습니다.")

    # 같은 prompt에는 같은 모델을 배정한다.
    prompt_groups: dict[str, list[int]] = {}
    for row_index, record in enumerate(routing_records):
        prompt_groups.setdefault(record["prompt"], []).append(row_index)
    for group in prompt_groups.values():
        group_choices = optimized.selected_indices[group]
        if np.any(group_choices != group_choices[0]):
            cheapest = min(
                (int(choice) for choice in np.unique(group_choices)),
                key=lambda choice: float(costs[group[0], choice]),
            )
            optimized.selected_indices[group] = cheapest

    selected_by_original = np.zeros(len(records), dtype=np.int64)
    for routing_index, original_index in enumerate(routing_order):
        selected_by_original[original_index] = optimized.selected_indices[routing_index]

    decisions = [
        {
            "episode_id": episode_id,
            "model_id": MODEL_NAMES[int(selected_by_original[index])],
        }
        for index, episode_id in enumerate(episode_ids)
    ]
    _validate_decisions(episode_ids, decisions)
    submission = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "challenge_id": challenge_id,
        "policy_id": POLICY_ID,
        "split": split,
        "tier": args.tier,
        "decisions": decisions,
    }
    write_json(args.output, submission)
    log = {
        "tier": args.tier,
        "decisions": len(decisions),
        "budget_ratio": selected_cost / light_baseline_cost,
    }
    print(json.dumps(log, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
