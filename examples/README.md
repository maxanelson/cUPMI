# Synthetic example

Run:

```bash
python examples/synthetic_demo.py
```

The script creates several synthetic probability streams and uses
`evaluate_over_seeds` to compare a plain random-forest stack with the same stack
wrapped in `CUPMICombiner`, over 5 seeds on fixed outer folds. It prints the
per-seed table, the summary over seeds (macro AUC and QWK), and how often each
`rho` was selected. It takes about 20 seconds.

The exact delta is not important; cUPMI is a regularizer, not a guarantee of
improvement on every synthetic draw. The example demonstrates the API, the
fold-locked evaluation protocol, and the selected synthesis ratios.
