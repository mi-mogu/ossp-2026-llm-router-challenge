# SPDX-FileCopyrightText: Copyright 2026 OSSP Router contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
COORDINATOR_DIR = ROOT / "coordinator"
NGRAM_DIR = ROOT / "n-gram"
ROUTER_DIR = ROOT / "라우터 분류"
NGRAM_MODEL = NGRAM_DIR / "ngram_classifier_v2_float32.joblib"
ROUTER_SCRIPT = ROUTER_DIR / "router.py"
DEFAULT_CHUNK_SIZE = 500
TIERS = ("fast", "balanced", "premium")
MODELS = ("ax31-light", "ax31", "axk1-think")
MESSAGE_ROLES = {"system", "user", "assistant"}

for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(key, "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OSSP prompt router")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--tier", required=True, choices=TIERS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    return parser.parse_args()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON에 중복 키가 있습니다: {key!r}")
        result[key] = value
    return result


def _read_input(path: Path) -> tuple[str, str, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"입력 JSON이 없습니다: {path}")
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file, object_pairs_hook=_reject_duplicate_keys)

    if not isinstance(data, Mapping):
        raise ValueError("공식 입력 JSON의 최상위 값은 object여야 합니다.")
    if set(data) != {"schema_version", "challenge_id", "split", "episodes"}:
        raise ValueError("입력 JSON의 최상위 필드가 v1 스키마와 다릅니다.")
    if data["schema_version"] != 1 or isinstance(data["schema_version"], bool):
        raise ValueError("schema_version은 정수 1이어야 합니다.")

    challenge_id = data["challenge_id"]
    split = data["split"]
    episodes = data["episodes"]
    if not isinstance(challenge_id, str) or not challenge_id.strip():
        raise ValueError("challenge_id는 비어 있지 않은 문자열이어야 합니다.")
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split은 비어 있지 않은 문자열이어야 합니다.")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("episodes는 비어 있지 않은 배열이어야 합니다.")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise ValueError(f"record {index}: JSON object가 아닙니다.")

        has_prompt = "prompt" in episode
        has_messages = "messages" in episode
        body = "prompt" if has_prompt else "messages"
        if has_prompt == has_messages or set(episode) != {"episode_id", body}:
            raise ValueError(
                f"record {index}: episode_id와 prompt/messages 중 정확히 하나만 허용됩니다."
            )

        episode_id = episode.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id or len(episode_id) > 128:
            raise ValueError(f"record {index}: episode_id는 1~128자 문자열이어야 합니다.")
        if episode_id in seen:
            raise ValueError(f"중복 episode_id가 있습니다: {episode_id!r}")
        seen.add(episode_id)

        if has_prompt:
            prompt = episode["prompt"]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"record {index}: prompt는 비어 있지 않은 문자열이어야 합니다.")
        else:
            messages = episode["messages"]
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"record {index}: messages는 비어 있지 않은 배열이어야 합니다.")
            parts: list[str] = []
            for message_index, message in enumerate(messages):
                if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
                    raise ValueError(
                        f"record {index}.messages[{message_index}]: role/content만 허용됩니다."
                    )
                role = message["role"]
                content = message["content"]
                if role not in MESSAGE_ROLES:
                    raise ValueError(
                        f"record {index}.messages[{message_index}]: 허용되지 않은 role입니다."
                    )
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(
                        f"record {index}.messages[{message_index}]: content가 비어 있습니다."
                    )
                parts.append(content)
            prompt = "\n".join(parts)

        records.append({"episode_id": episode_id, "prompt": prompt})
    return challenge_id, split, records


def _classify_all_chunks(
    records: list[dict[str, Any]], chunk_size: int
) -> list[dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("--chunk-size는 1 이상이어야 합니다.")

    chunk_path = COORDINATOR_DIR / "chunk_processor.py"
    ngram_path = NGRAM_DIR / "ngram_predict_optimized.py"
    for path in (chunk_path, ngram_path, NGRAM_MODEL):
        if not path.is_file():
            raise FileNotFoundError(f"실행 파일이 없습니다: {path}")

    chunks = _load_module("router_chunks", chunk_path)
    ngram = _load_module("router_ngram", ngram_path)
    ngram.initialize(NGRAM_DIR, NGRAM_MODEL)

    def classify(chunk: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        predictions = ngram.predict([item["prompt"] for item in chunk])
        if len(predictions) != len(chunk):
            raise ValueError("n-gram 결과 수가 입력 수와 다릅니다.")

        answer: list[dict[str, Any]] = []
        for item, prediction in zip(chunk, predictions):
            label = prediction.get("prediction")
            if not isinstance(label, Mapping):
                raise ValueError("n-gram 결과에 prediction object가 없습니다.")
            row = {
                "episode_id": item["episode_id"],
                "prompt": item["prompt"],
                "label": dict(label),
            }
            probabilities = prediction.get("probabilities")
            if isinstance(probabilities, Mapping):
                row["classification_probabilities"] = dict(probabilities)
            answer.append(row)
        return answer

    return chunks.run_chunked(records, classify, chunk_size, validate_episode_ids=True)


def _validate_submission(
    submission: Any,
    tier: str,
    input_ids: list[Any],
    challenge_id: str,
    split: str,
) -> None:
    if not isinstance(submission, Mapping):
        raise ValueError("최종 결과가 JSON object가 아닙니다.")
    if set(submission) != {
        "schema_version",
        "challenge_id",
        "policy_id",
        "split",
        "tier",
        "decisions",
    }:
        raise ValueError("최종 결과의 최상위 필드가 제출 스키마와 다릅니다.")
    if submission["schema_version"] != 1:
        raise ValueError("schema_version이 1이 아닙니다.")
    if submission["challenge_id"] != challenge_id:
        raise ValueError("challenge_id가 입력과 다릅니다.")
    if submission["policy_id"] != "ossp-2026-prompt-router-v1":
        raise ValueError("policy_id가 올바르지 않습니다.")
    if submission["split"] != split:
        raise ValueError("split이 입력과 다릅니다.")
    if submission["tier"] != tier:
        raise ValueError("tier가 요청과 다릅니다.")

    decisions = submission["decisions"]
    if not isinstance(decisions, list) or len(decisions) != len(input_ids):
        raise ValueError("decisions 수가 입력 episode 수와 다릅니다.")

    output_ids = []
    for index, decision in enumerate(decisions):
        if not isinstance(decision, Mapping) or set(decision) != {
            "episode_id",
            "model_id",
        }:
            raise ValueError(f"decision {index}: 제출 필드가 올바르지 않습니다.")
        if decision["model_id"] not in MODELS:
            raise ValueError(f"decision {index}: 허용되지 않은 model_id입니다.")
        output_ids.append(decision["episode_id"])
    if output_ids != input_ids:
        raise ValueError("최종 episode_id 전체 또는 순서가 입력과 다릅니다.")


def _write_submission_atomic(output: Path, submission: Mapping[str, Any]) -> None:
    """Publish a complete submission in one replace operation."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(submission, file, ensure_ascii=False, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, output)
        os.chmod(output, 0o644)
    finally:
        temporary.unlink(missing_ok=True)


def _write_light_fallback(
    output: Path,
    tier: str,
    challenge_id: str,
    split: str,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Emit a valid, budget-minimal submission after an internal pipeline failure."""
    submission = {
        "schema_version": 1,
        "challenge_id": challenge_id,
        "policy_id": "ossp-2026-prompt-router-v1",
        "split": split,
        "tier": tier,
        "decisions": [
            {"episode_id": record["episode_id"], "model_id": "ax31-light"}
            for record in records
        ],
    }
    _validate_submission(
        submission,
        tier,
        [record["episode_id"] for record in records],
        challenge_id,
        split,
    )
    _write_submission_atomic(output, submission)


def _run_router(
    records: list[dict[str, Any]],
    tier: str,
    directory: Path,
    challenge_id: str,
    split: str,
) -> tuple[Path, dict[str, Any]]:
    if not ROUTER_SCRIPT.is_file():
        raise FileNotFoundError(f"최종 모델 선택기가 없습니다: {ROUTER_SCRIPT}")

    input_path = directory / "classified.json"
    output_path = directory / "submission.json"
    with input_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(
            {
                "schema_version": 1,
                "challenge_id": challenge_id,
                "split": split,
                "episodes": records,
            },
            file,
            ensure_ascii=False,
            allow_nan=False,
        )

    command = [
        sys.executable,
        str(ROUTER_SCRIPT),
        "--input",
        str(input_path),
        "--tier",
        tier,
        "--output",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    with output_path.open("r", encoding="utf-8-sig") as file:
        return output_path, json.load(file)


def run_pipeline(args: argparse.Namespace) -> int:
    challenge_id, split, records = _read_input(args.input.resolve())
    classified = _classify_all_chunks(records, args.chunk_size)
    if len(classified) != len(records):
        raise RuntimeError("전체 n-gram 결과 수가 입력 수와 다릅니다.")

    with tempfile.TemporaryDirectory(prefix="prompt_router_") as name:
        _router_output, submission = _run_router(
            classified,
            args.tier,
            Path(name),
            challenge_id,
            split,
        )
        _validate_submission(
            submission,
            args.tier,
            [record["episode_id"] for record in records],
            challenge_id,
            split,
        )

        _write_submission_atomic(args.output, submission)

    print(f"tier={args.tier} decisions={len(records)} output={args.output.resolve()}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run_pipeline(args)
    except Exception as error:
        # Official input validation remains strict. If the valid input can be read,
        # recover from an internal classifier/model/optimizer failure with the
        # cheapest valid policy instead of losing the entire tier.
        try:
            challenge_id, split, records = _read_input(args.input.resolve())
            _write_light_fallback(
                args.output, args.tier, challenge_id, split, records
            )
        except Exception as fallback_error:
            print(f"실패: {error}; 안전 복구 실패: {fallback_error}", file=sys.stderr)
            return 2
        print(
            f"경고: 내부 실행 실패로 all-light 안전 복구를 적용했습니다: {error}",
            file=sys.stderr,
        )
        print(
            f"tier={args.tier} decisions={len(records)} "
            f"output={args.output.resolve()} fallback=all-light"
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
