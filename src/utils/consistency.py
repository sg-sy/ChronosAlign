"""Utilities for measuring temporal QA consistency."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def _stringify_answer(answer: Any) -> str:
    """Convert answer values to stable strings for deduplication."""
    if isinstance(answer, np.ndarray):
        return ",".join(map(str, answer.tolist()))
    if isinstance(answer, list):
        return " ".join(map(str, answer))
    return str(answer)


def measure_answer_consistency(
    question: str,
    df: pd.DataFrame,
    generated_answer_column: str = "generated_answer",
) -> Optional[float]:
    """
    Measure answer consistency for a given question across years.

    Returns a score in [0, 1):
    - Higher score means answers are more consistent.
    - Lower score means answers vary more across rows.
    """
    question_data = df[df["question"] == question]
    if len(question_data) == 0:
        return None

    answers = (
        question_data[generated_answer_column]
        .dropna()
        .apply(_stringify_answer)
        .tolist()
    )
    if len(answers) == 0:
        return None

    unique_answers = len(set(answers))
    total_answers = len(answers)
    return 1 - (unique_answers / total_answers)
