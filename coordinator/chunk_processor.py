# SPDX-FileCopyrightText: Copyright 2026 OSSP Router contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from itertools import islice
from typing import Any


DEFAULT_CHUNK_SIZE = 500

Sample = Mapping[str, Any]
ChunkProcessor = Callable[[Sequence[Sample]], Iterable[Any]]


def iter_chunks(items: Iterable[Sample], chunk_size: int) -> Iterator[list[Sample]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    iterator = iter(items)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk


def _extract_episode_ids(
    values: Sequence[Any],
    *,
    value_name: str,
    chunk_index: int,
) -> list[Any] | None:
    flags = [isinstance(value, Mapping) and "episode_id" in value for value in values]
    if not any(flags):
        return None
    if not all(flags):
        raise ValueError(
            f"chunk {chunk_index}: {value_name} contains only partial episode_id fields"
        )
    return [value["episode_id"] for value in values]


def _check_new_ids(
    episode_ids: Sequence[Any],
    seen_ids: set[Any],
    *,
    value_name: str,
    chunk_index: int,
) -> None:
    chunk_ids: set[Any] = set()
    for episode_id in episode_ids:
        try:
            duplicate = episode_id in chunk_ids or episode_id in seen_ids
        except TypeError as error:
            raise ValueError(
                f"chunk {chunk_index}: {value_name} episode_id must be hashable"
            ) from error
        if duplicate:
            raise ValueError(
                f"chunk {chunk_index}: duplicate {value_name} episode_id: {episode_id!r}"
            )
        chunk_ids.add(episode_id)
    seen_ids.update(chunk_ids)


def _validate_chunk_result(
    chunk: Sequence[Sample],
    chunk_results: Sequence[Any],
    *,
    chunk_index: int,
    validate_episode_ids: bool,
    seen_input_ids: set[Any],
    seen_output_ids: set[Any],
) -> None:
    if len(chunk_results) != len(chunk):
        raise ValueError(
            f"chunk {chunk_index}: process_chunk returned "
            f"{len(chunk_results)} results for {len(chunk)} inputs"
        )
    if not validate_episode_ids:
        return

    input_ids = _extract_episode_ids(
        chunk, value_name="input", chunk_index=chunk_index
    )
    if input_ids is None:
        return

    output_ids = _extract_episode_ids(
        chunk_results, value_name="output", chunk_index=chunk_index
    )
    if output_ids is None:
        raise ValueError(
            f"chunk {chunk_index}: outputs must include episode_id when "
            "episode-ID validation is enabled"
        )

    _check_new_ids(
        input_ids,
        seen_input_ids,
        value_name="input",
        chunk_index=chunk_index,
    )
    _check_new_ids(
        output_ids,
        seen_output_ids,
        value_name="output",
        chunk_index=chunk_index,
    )
    if input_ids != output_ids:
        raise ValueError(
            f"chunk {chunk_index}: process_chunk did not preserve episode_id order"
        )


def iter_processed_chunks(
    items: Iterable[Sample],
    process_chunk: ChunkProcessor,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    *,
    validate_episode_ids: bool = True,
) -> Iterator[tuple[int, list[Any]]]:
    if not callable(process_chunk):
        raise TypeError("process_chunk must be callable")

    seen_input_ids: set[Any] = set()
    seen_output_ids: set[Any] = set()
    for index, chunk in enumerate(iter_chunks(items, chunk_size)):
        returned = process_chunk(chunk)
        if returned is None:
            raise ValueError(f"chunk {index}: process_chunk returned None")
        try:
            results = list(returned)
        except TypeError as error:
            raise ValueError(
                f"chunk {index}: process_chunk must return an iterable"
            ) from error

        _validate_chunk_result(
            chunk,
            results,
            chunk_index=index,
            validate_episode_ids=validate_episode_ids,
            seen_input_ids=seen_input_ids,
            seen_output_ids=seen_output_ids,
        )
        yield index, results


def run_chunked(
    items: Iterable[Sample],
    process_chunk: ChunkProcessor,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    *,
    validate_episode_ids: bool = True,
) -> list[Any]:
    answer: list[Any] = []
    for _, results in iter_processed_chunks(
        items,
        process_chunk,
        chunk_size,
        validate_episode_ids=validate_episode_ids,
    ):
        answer.extend(results)
    return answer
