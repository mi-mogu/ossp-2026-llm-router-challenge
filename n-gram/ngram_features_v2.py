# SPDX-FileCopyrightText: Copyright 2026 OSSP Router contributors
# SPDX-License-Identifier: Apache-2.0

import math
import re

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


NUMBER_PATTERN = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:[.,:/-]\d+)*(?![A-Za-z_])")
WHITESPACE_PATTERN = re.compile(r"\s+")
MULTIPLE_CHOICE_PATTERN = re.compile(
    r"(?:^|\s)(?:[A-Ea-e]|[1-5])[.)]\s|①|②|③|④|⑤",
    re.MULTILINE,
)
CODE_WORD_PATTERN = re.compile(
    r"\b(?:def|class|return|import|function|assert|println|console|select|from)\b",
    re.IGNORECASE,
)


class PromptPreprocessor:
    def __init__(
        self,
        lowercase=True,
        normalize_whitespace=False,
        mask_numbers=False,
        max_chars=None,
    ):
        self.lowercase = lowercase
        self.normalize_whitespace = normalize_whitespace
        self.mask_numbers = mask_numbers
        self.max_chars = max_chars

    def __call__(self, text):
        if self.max_chars and len(text) > self.max_chars:
            head_size = self.max_chars // 2
            tail_size = self.max_chars - head_size
            text = text[:head_size] + "\n<TRUNCATED>\n" + text[-tail_size:]

        if self.mask_numbers:
            text = NUMBER_PATTERN.sub(" <NUM> ", text)

        if self.normalize_whitespace:
            text = WHITESPACE_PATTERN.sub(" ", text).strip()

        if self.lowercase:
            text = text.lower()

        return text


class PromptStatsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor

    def fit(self, values, y=None):
        return self

    def transform(self, values):
        rows = []

        for original_text in values:
            text = (
                self.preprocessor(original_text)
                if self.preprocessor is not None
                else original_text
            )
            length = max(len(text), 1)
            line_count = text.count("\n") + 1
            hangul_count = sum("가" <= char <= "힣" for char in text)
            latin_count = sum(char.isascii() and char.isalpha() for char in text)
            digit_count = sum(char.isdigit() for char in text)
            whitespace_count = sum(char.isspace() for char in text)
            punctuation_count = sum(
                not char.isalnum() and not char.isspace()
                for char in text
            )
            bracket_count = sum(char in "()[]{}<>" for char in text)
            quote_count = sum(char in "'\"`" for char in text)
            multiple_choice_count = len(MULTIPLE_CHOICE_PATTERN.findall(text))
            code_word_count = len(CODE_WORD_PATTERN.findall(text))

            rows.append(
                [
                    min(math.log1p(length) / math.log1p(70000), 1.5),
                    min(math.log1p(line_count) / math.log1p(5000), 1.5),
                    hangul_count / length,
                    latin_count / length,
                    digit_count / length,
                    whitespace_count / length,
                    punctuation_count / length,
                    bracket_count / length,
                    quote_count / length,
                    min(multiple_choice_count / 10, 1.0),
                    min(code_word_count / 10, 1.0),
                    min(text.count("?") / 5, 1.0),
                    float("```" in text),
                    float("assert" in text.lower()),
                ]
            )

        return sparse.csr_matrix(np.asarray(rows, dtype=np.float64))


def make_preprocessor(config):
    preprocessing = config.get("preprocessing", {})

    return PromptPreprocessor(
        lowercase=preprocessing.get("lowercase", True),
        normalize_whitespace=preprocessing.get("normalize_whitespace", False),
        mask_numbers=preprocessing.get("mask_numbers", False),
        max_chars=preprocessing.get("max_chars"),
    )


def make_tfidf(name, tfidf_config, preprocessor):
    return (
        name,
        TfidfVectorizer(
            analyzer=tfidf_config["analyzer"],
            ngram_range=tuple(tfidf_config["ngram_range"]),
            min_df=tfidf_config.get("min_df", 2),
            max_df=tfidf_config.get("max_df", 1.0),
            max_features=tfidf_config.get("max_features"),
            sublinear_tf=tfidf_config.get("sublinear_tf", True),
            binary=tfidf_config.get("binary", False),
            norm=tfidf_config.get("norm", "l2"),
            preprocessor=preprocessor,
            lowercase=False,
        ),
    )


def make_feature_extractor(config):
    preprocessor = make_preprocessor(config)
    transformers = []
    weights = {}

    if config.get("char"):
        transformers.append(make_tfidf("char", config["char"], preprocessor))
        weights["char"] = config.get("char_weight", 1.0)

    if config.get("word"):
        transformers.append(make_tfidf("word", config["word"], preprocessor))
        weights["word"] = config.get("word_weight", 1.0)

    stats_weight = config.get("stats_weight", 0.0)

    if stats_weight > 0:
        transformers.append(
            (
                "stats",
                PromptStatsTransformer(preprocessor=preprocessor),
            )
        )
        weights["stats"] = stats_weight

    if len(transformers) == 1 and transformers[0][0] != "stats":
        return transformers[0][1]

    return FeatureUnion(
        transformers,
        transformer_weights=weights,
    )
