# Contributing to pyTransitPhotometry

Thank you for your interest in contributing to `pyTransitPhotometry`. This document describes how to report bugs, propose features, and submit code contributions.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Reporting Issues](#reporting-issues)
- [Requesting Features](#requesting-features)
- [Development Setup](#development-setup)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating you agree to abide by its terms. Report unacceptable behaviour to the maintainer via a GitHub issue marked **[private]** or by email.

---

## Reporting Issues

Please use the **Bug Report** issue template on GitHub. Include:

1. A minimal reproducible example (small synthetic FITS data or a unit test snippet).
2. The full error traceback.
3. Your platform, Python version, and key package versions (`photutils`, `batman-package`, `astropy`).
4. What you expected to happen and what actually happened.

---

## Requesting Features

Open a **Feature Request** issue with:

1. A clear description of the scientific motivation (what analysis does this enable?).
2. Any relevant references (papers, other packages).
3. A sketch of the proposed API or configuration schema change.

---

## Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/CuriousAvenger/pyTransitPhotometry.git
cd pyTransitPhotometry

# 2. Create an isolated environment
conda create -n pytransit-dev python=3.11
conda activate pytransit-dev

# 3. Install in editable mode with dev extras
pip install -e ".[dev]"
```

The `[dev]` extras install `pytest`, `pytest-cov`, `black`, and `flake8`.

---

## Submitting a Pull Request

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```
2. Make your changes. Keep each commit focused on one logical change.
3. Add or update tests in `tests/` so that `pytest` coverage does not decrease.
4. Run the full test suite locally:
   ```bash
   pytest tests/ -v --tb=short
   ```
5. Format your code:
   ```bash
   black pyTransitPhotometry/ tests/
   flake8 pyTransitPhotometry/ tests/ --max-line-length=100
   ```
6. Push your branch and open a pull request against `main`. Fill in the PR template explaining what changed and why.
7. Address reviewer comments in subsequent commits on the same branch; do not force-push after review has started.

---

## Coding Standards

- **Style**: [Black](https://black.readthedocs.io/) (line length 100).
- **Linting**: [Flake8](https://flake8.pycqa.org/) with `--max-line-length=100`.
- **Type hints**: Encouraged for all public functions and class methods.
- **Docstrings**: NumPy style. Every public function must have a summary line, `Parameters`, `Returns`, and at least one `Examples` entry.
- **No commented-out code** in submitted PRs.

---

## Testing

The test suite lives in `tests/test_pipeline.py` and is run with `pytest`. All tests must pass before a PR is merged:

```bash
pytest tests/ -v
```

For coverage reporting:

```bash
pytest tests/ --cov=pyTransitPhotometry --cov-report=term-missing
```

New features **must** include unit tests. Bug fixes **must** include a regression test that would have caught the bug.

---

## Documentation

User-facing documentation lives in `docs/`. If you add a new public function or configuration key, update the relevant page in `docs/`. If you add a new pipeline stage, update `docs/usage.md`.

For significant changes, consider adding a worked example to `examples/tutorial.ipynb`.

---

## Getting Help

Open a [GitHub Discussion](https://github.com/CuriousAvenger/pyTransitPhotometry/discussions) or file an issue tagged `[question]`.
