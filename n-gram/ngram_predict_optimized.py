# SPDX-FileCopyrightText: Copyright 2026 OSSP Router contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


_BUNDLE = None
_PREDICT_MODULE = None


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def initialize(ngram_dir, model_path=None):
    global _BUNDLE, _PREDICT_MODULE
    if _BUNDLE is not None:
        return
    root = Path(ngram_dir).resolve()
    sys.path.insert(0, str(root))
    import joblib

    _PREDICT_MODULE = _load_module(
        "predict_ngram_v2_worker", root / "predict_ngram_v2.py"
    )
    path = Path(model_path).resolve() if model_path else root / "ngram_classifier_v2.joblib"
    _BUNDLE = joblib.load(path)


def predict(prompts):
    return _PREDICT_MODULE.predict(_BUNDLE, prompts)


def _worker_initialize(ngram_dir, model_path):
    initialize(ngram_dir, model_path)


def _worker_predict(prompts):
    return predict(prompts)


def _split_contiguous(values, parts):
    quotient, remainder = divmod(len(values), parts)
    chunks, start = [], 0
    for index in range(parts):
        size = quotient + int(index < remainder)
        if size:
            chunks.append(values[start : start + size])
        start += size
    return chunks


class ParallelNgramPredictor:
    def __init__(self, ngram_dir, workers, model_path=None):
        self.workers = workers
        self.executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_initialize,
            initargs=(
                str(Path(ngram_dir).resolve()),
                str(Path(model_path).resolve()) if model_path else None,
            ),
        )

    def predict(self, prompts):
        chunks = _split_contiguous(prompts, min(self.workers, len(prompts)))
        nested = self.executor.map(_worker_predict, chunks)
        return [item for chunk in nested for item in chunk]

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
