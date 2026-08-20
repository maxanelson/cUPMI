# Contributing

Thanks for improving cUPMI. This project is intentionally small and generic.

Before opening a pull request:

1. Keep examples synthetic or publicly reproducible.
2. Do not add real data, trained artifacts, private paths, or run logs.
3. Add or update tests for behavior changes.
4. Run:

```bash
pip install -e ".[dev]"
pytest
```

Core code should remain compatible with scikit-learn estimators that implement
`fit` and `predict_proba`.
