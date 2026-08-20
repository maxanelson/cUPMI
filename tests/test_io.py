# SPDX-License-Identifier: Apache-2.0

import json

import pandas as pd

from cupmi.io import read_probability_csv, read_split_json


def test_read_probability_csv(tmp_path):
    path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "prob_0": [0.8, 0.2],
            "prob_1": [0.2, 0.8],
        }
    ).to_csv(path, index=False)

    ids, proba = read_probability_csv(path)

    assert ids.tolist() == ["a", "b"]
    assert proba.shape == (2, 2)


def test_read_split_json(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            {
                "folds": [
                    {"fold": 0, "test": [{"sample_id": "a"}, {"sample_id": "b"}]},
                    {"fold": 1, "test": ["c"]},
                ]
            }
        )
    )

    assert read_split_json(path) == {"a": 0, "b": 0, "c": 1}
