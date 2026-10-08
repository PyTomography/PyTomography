# Contributing to PyTomography

PyTomography is built by the people who use it. The full guide is in the documentation under
[Contributing](https://pytomography.readthedocs.io/en/latest/contributing/index.html), and the
[feature board](https://pytomography.readthedocs.io/en/latest/contributing/feature-board.html) lists work we would like
done, each item with data to test on, a place in the code to start, and a definition of done.

## Quick start

```bash
git clone https://github.com/<your-username>/PyTomography.git
cd PyTomography
pip install -e ".[dev]"
pre-commit install
pytest
```

A GPU is not needed: tests that need one are skipped, and so are tests that need the tutorial data unless
`PYTOMOGRAPHY_DATA` points at it. The PET and CT projector tests need parallelproj 2, which comes from conda-forge:

```bash
conda env create -f .github/ci-environment.yml
conda activate pytomography-ci
pip install -e ".[test]"
pytest
```

## What runs on your pull request

Every push runs every test that doesn't need a GPU, on GitHub's runners: with pip on Linux, macOS and Windows and
Python 3.10 to 3.13; with parallelproj 2 from conda on all three systems; the regression tests on tutorial data; a
check that the scripts in `examples/` match the tutorial notebooks; and a check that the pull request closes an issue.

When a maintainer approves your pull request, they add it to the merge queue, which runs those checks again on the
exact commit that would land on `main` and merges only if they pass. GitHub's runners have no GPU, so the few tests
that need one skip themselves there; maintainers run them on a GPU machine.

## How changes get in

1. Every change starts from an issue. Comment on the one you want to work on, or open one, so a maintainer can point
   you to the right place.
2. Branch from `main`. It is the only long-lived branch; releases are tags on it.
3. Add a test that fails before your change. A new system matrix or algorithm should inherit the contract tests in
   `src/pytomography/tests/contracts.py`.
4. Open a pull request against `main` that says which issue it closes, e.g. `Closes #123` in its description, and
   fill in the template. A check fails a pull request that closes no issue; for a trivial change, such as a typo, a
   maintainer can waive it with the `no-issue` label. Every pull request gets a first reply within five working days.

Contributors are credited in the release notes, and substantial features earn co-authorship on the next
PyTomography paper.
