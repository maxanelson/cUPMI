# Input Contracts

The core API accepts NumPy arrays directly. File readers are optional helpers for
projects that prefer a lightweight CSV/JSON convention.

## Probability stream CSV

One row per sample, one probability column per class.

```text
sample_id,prob_0,prob_1,prob_2
s0001,0.80,0.15,0.05
s0002,0.10,0.70,0.20
```

Use:

```python
from cupmi.io import read_probability_csv

ids, probabilities = read_probability_csv("stream_a_predictions.csv")
```

Probability streams must be aligned to the same sample order before calling
`stack_log_proba` or `evaluate_precomputed_streams`.

## Split JSON

The split file maps sample identifiers to fixed outer folds.

```json
{
  "folds": [
    {"fold": 0, "test": [{"sample_id": "s0001"}, {"sample_id": "s0002"}]},
    {"fold": 1, "test": [{"sample_id": "s0003"}]}
  ]
}
```

String-only rows are also accepted:

```json
{"folds": [{"fold": 0, "test": ["s0001", "s0002"]}]}
```

Use:

```python
from cupmi.io import read_split_json

fold_of = read_split_json("split.json")
```

## Array-first workflow

For most projects, the cleanest workflow is:

```python
from cupmi import stack_log_proba, CUPMICombiner

U = stack_log_proba([prob_stream_1, prob_stream_2, prob_stream_3])
clf = CUPMICombiner(estimator="rf")
clf.fit(U_train, y_train)
```

The package does not assume any domain-specific preprocessing, image format, or
feature-extraction tool.
