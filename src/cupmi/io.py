# SPDX-License-Identifier: Apache-2.0
"""Optional readers for generic cUPMI input contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_probability_csv(
    path: str | Path,
    *,
    id_col: str = "sample_id",
    prob_prefix: str = "prob_",
) -> tuple[np.ndarray, np.ndarray]:
    """Read a probability CSV as ``(sample_ids, probabilities)``.

    The CSV must contain one identifier column plus columns named like
    ``prob_0``, ``prob_1``, ...
    """

    df = pd.read_csv(path)
    if id_col not in df.columns:
        raise ValueError(f"Missing identifier column: {id_col!r}")
    prob_cols = sorted(col for col in df.columns if col.startswith(prob_prefix))
    if len(prob_cols) < 2:
        raise ValueError(f"Expected at least two columns with prefix {prob_prefix!r}.")
    return df[id_col].astype(str).to_numpy(), df[prob_cols].to_numpy(float)


def read_split_json(path: str | Path, *, id_key: str = "sample_id") -> dict[str, int]:
    """Read a split JSON into ``{sample_id: fold}``.

    Supported shape:

    ``{"folds": [{"fold": 0, "test": [{"sample_id": "a"}, ...]}, ...]}``

    ``"val"`` is also accepted as an alias for ``"test"``.
    """

    raw = json.loads(Path(path).read_text())
    fold_of: dict[str, int] = {}
    for fold_record in raw.get("folds", []):
        fold = int(fold_record["fold"])
        rows = fold_record.get("test", fold_record.get("val", []))
        for row in rows:
            if isinstance(row, str):
                sample_id = row
            else:
                sample_id = str(row[id_key])
            fold_of[sample_id] = fold
    if not fold_of:
        raise ValueError("No fold assignments found.")
    return fold_of
