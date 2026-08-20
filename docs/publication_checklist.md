# Publication Checklist

Run this checklist before creating a public repository, release archive, or
package upload.

## Candidate directory

Work from a fresh directory with fresh git history. Do not publish a filtered
copy of a private research workspace.

## Required scans

From the candidate repository root:

```bash
rg -n "/Users/|/home/|/Volumes/|DataRoom|server|cluster|ssh|token|password|secret" . \
  --glob '!docs/publication_checklist.md'
```

Expected result: no private paths, host names, or credentials.

```bash
rg -n "\\b(MRN|DOB|gender|sex|age_at|accession|hospital|site_code)\\b" . \
  --glob '!docs/publication_checklist.md'
```

Expected result: zero hits unless they are in this checklist.

```bash
find . -type f \\( -name "*.csv" -o -name "*.jsonl" -o -name "*.nii" -o -name "*.nii.gz" -o -name "*.pt" -o -name "*.pth" -o -name "*.joblib" \\)
```

Expected result: zero real data files or trained artifacts. Tiny synthetic test
fixtures are allowed only under `tests/fixtures/` and must be clearly named.

```bash
find . -size +1M -type f -print
```

Expected result: no large files except ordinary package metadata if generated
locally and excluded from git.

## Functional checks

```bash
pip install -e ".[dev]"
pytest
python examples/synthetic_demo.py
```

## Git checks

```bash
git status --short
git log --oneline --decorate --max-count=5
```

Expected result: fresh history containing only public-package commits.

## Optional external scanners

If available:

```bash
gitleaks detect --no-git
trufflehog filesystem .
```

The release should contain only generic source code, documentation, tests, and
synthetic examples.
