from __future__ import annotations

from collections.abc import Sequence


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref_char in enumerate(reference, start=1):
        current = [row]
        for column, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1]


def sequence_metrics(references: Sequence[str], hypotheses: Sequence[str]) -> dict[str, float | int]:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have equal length")
    total_edits = 0
    total_characters = 0
    exact = 0
    similarities: list[float] = []
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        distance = edit_distance(reference, hypothesis)
        total_edits += distance
        total_characters += len(reference)
        exact += int(reference == hypothesis)
        denominator = max(len(reference), len(hypothesis), 1)
        similarities.append(max(0.0, 1.0 - distance / denominator))
    count = len(references)
    return {
        "samples": count,
        "character_error_rate": total_edits / max(total_characters, 1),
        "mean_normalized_edit_similarity": sum(similarities) / max(count, 1),
        "exact_sequence_match": exact / max(count, 1),
    }
