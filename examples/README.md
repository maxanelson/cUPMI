# Synthetic example

Run:

```bash
python examples/synthetic_demo.py
```

The script creates several synthetic probability streams, converts them to
stacked log-probability meta-features, and compares a plain random-forest stack
with the same stack wrapped in `CUPMICombiner`.

The exact delta is not important; cUPMI is a regularizer, not a guarantee of
improvement on every synthetic draw. The example demonstrates the API, the
fold-locked evaluation protocol, and the selected synthesis ratios.
