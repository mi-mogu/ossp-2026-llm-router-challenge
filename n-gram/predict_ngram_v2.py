# SPDX-FileCopyrightText: Copyright 2026 OSSP Router contributors
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "ngram_classifier_v2.joblib"
AXES = [
    "response_language",
    "request_types",
    "content_domains",
    "answer_formats",
]


def probabilities_by_label(encoder, probabilities):
    return {
        str(label): float(probability)
        for label, probability in zip(encoder.classes_, probabilities)
    }


def predict(bundle, prompts):
    features = bundle["feature_extractor"].transform(prompts)
    results = []

    axis_probabilities = {
        axis: np.asarray(bundle["classifiers"][axis].predict_proba(features))
        for axis in AXES
    }

    for row_index, prompt in enumerate(prompts):
        output = {"prompt": prompt, "prediction": {}, "probabilities": {}}

        for axis in AXES:
            encoder = bundle["encoders"][axis]
            probabilities = axis_probabilities[axis][row_index]
            output["probabilities"][axis] = probabilities_by_label(
                encoder,
                probabilities,
            )

            if bundle["thresholds"][axis] is None:
                label_index = int(np.argmax(probabilities))
                label = str(encoder.classes_[label_index])
                output["prediction"][axis] = label
            else:
                selected = [
                    str(label)
                    for label, probability in zip(encoder.classes_, probabilities)
                    if probability >= bundle["thresholds"][axis][str(label)]
                ]
                if not selected:
                    selected = [
                        str(encoder.classes_[int(np.argmax(probabilities))])
                    ]
                output["prediction"][axis] = selected

        results.append(output)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="*")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input-jsonl", type=Path)
    args = parser.parse_args()

    prompts = list(args.prompt)
    if args.input_jsonl:
        with args.input_jsonl.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    prompts.append(json.loads(line)["prompt"])
    if not prompts:
        raise ValueError("분류할 prompt를 한 개 이상 입력하세요.")

    started = time.perf_counter()
    bundle = joblib.load(args.model)
    print(json.dumps(predict(bundle, prompts), ensure_ascii=False, indent=2))
    print(f"소요시간: {time.perf_counter() - started:.5f} 초")


if __name__ == "__main__":
    main()
