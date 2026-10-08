# Contributing to PyTomography

PyTomography is built by the people who use it. The full guide is in the documentation under
[Contributing](https://pytomography.readthedocs.io/en/latest/contributing/index.html), and the
[feature board](https://pytomography.readthedocs.io/en/latest/contributing/feature-board.html) lists work we would like
done, each item with data to test on, a place in the code to start, and a definition of done. Questions about using
PyTomography go to [GitHub Discussions](https://github.com/PyTomography/PyTomography/discussions/categories/q-a).

## Quick start

```bash
git clone https://github.com/<your-username>/PyTomography.git
cd PyTomography
git checkout development
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

## The branches

| Branch | What it holds | Pull requests from |
|---|---|---|
| `development` | the next release; the default branch | your feature or fix branch |
| `main` | released code only; every release is a tag on it (`v4.0.0`) | `development` (a release) or a `hotfix/` branch |

Branch from `development` and open your pull request into `development`. Name the branch after what it does, e.g.
`fix/spect-rescale-slope` or `feature/kem-system-matrix`.

## How changes get in

1. Every change starts from an issue. Comment on the one you want to work on, or open one, so a maintainer can point
   you to the right place.
2. Branch from `development`, and add a test that fails before your change. A new system matrix or algorithm should
   inherit the contract tests in `src/pytomography/tests/contracts.py`.
3. Open a pull request into `development` that says which issue it closes, e.g. `Closes #123` in its description, and
   fill in the template. For a trivial change, such as a typo, a maintainer can waive the issue with the `no-issue`
   label. Every pull request gets a first reply within five working days.
4. Before it can merge, a pull request needs:
   - **a linked issue** (`Closes #123`), checked automatically;
   - **green checks** (see below);
   - **every review conversation resolved**.
5. A maintainer reviews your pull request and merges it with **squash and merge**: the pull request becomes one commit
   on `development`, whose message is the pull request's title and description. Write the title as a sentence that
   will read well in the history, and keep `Closes #123` in the description. The branch is deleted after the merge.

Nothing reaches `development` or `main` except through a pull request. Maintainers may merge their own pull requests
once the checks pass; GitHub still asks the code owners to review them. Approvals are not required for now, and may be
required later as more maintainers review. If a required check breaks for a reason that has nothing to do with the
pull request, such as an outage, an admin can bypass it, but only when merging a pull request, so the bypass is
recorded there.

## What runs on your pull request

Every push runs every test that doesn't need a GPU, on GitHub's runners: with pip on Linux, macOS and Windows and
Python 3.10 to 3.13; with parallelproj 2 from conda on all three systems; the regression tests on tutorial data; a
check that the scripts in `examples/` match the tutorial notebooks; and a check that the pull request closes an issue.
The required check "All tests passed" waits for all of these. GitHub's runners have no GPU, so the few tests that need
one skip themselves there; maintainers run them on a GPU machine.

## Stacked pull requests

When one change builds on another that hasn't merged yet, open the second pull request into the first one's branch
instead of into `development`, and say so at the top of its description (`Stacked on #241`). Review them in order.

Merge them bottom first. When the bottom pull request is squash-merged and its branch is deleted, GitHub retargets the
next one to `development` by itself. Its branch still carries the bottom branch's original commits, which now exist on
`development` only as the squashed commit, so rebase it onto `development` before it merges:

```bash
git fetch origin
# <old-base> is the commit the bottom branch pointed to when you branched from it (git log shows it)
git rebase --onto origin/development <old-base> my-stacked-branch
git push --force-with-lease
```

The rules only apply to `development` and `main`, so a stacked pull request can't land anywhere by accident: each one
reaches `development` with its own review, checks and squash commit.

## Releases and hotfixes

- **A release** is a pull request from `development` into `main`, merged with a **merge commit** (not squashed, so
  `main` and `development` share history). It closes the release's tracking issue. A maintainer then tags the merge
  commit on `main` (`git tag -a v4.1.0 -m "PyTomography 4.1.0"` and `git push origin v4.1.0`), and the tag publishes
  the package to PyPI. Only admins can create `v*` tags, and nobody can move or delete one.
- **A hotfix** for a released version branches from `main` as `hotfix/<what-it-fixes>`, goes into `main` through a
  pull request with the same rules (merge commit), and is tagged as a patch release. Then a pull request from `main`
  into `development`, merged with a merge commit, brings the fix back.
- A check fails any pull request into `main` that comes from anywhere other than `development` or a `hotfix/` branch.

Release notes are generated from pull request labels: `breaking change`, `enhancement`, `bug`, `documentation` and
`dependencies` each get a section, and `skip release notes` leaves a pull request out.

## Versions and deprecations

PyTomography follows [semantic versioning](https://semver.org): `MAJOR.MINOR.PATCH`.

- **Patch** (4.0.1): bug fixes only.
- **Minor** (4.1.0): new features; existing code keeps working.
- **Major** (5.0.0): may remove or change public functions and classes.

To remove or change something public, deprecate it first: keep it working for at least one minor release, with a
`DeprecationWarning` that says what to use instead and the version it will be removed in, then remove it in the next
major release. Label the pull request that removes it `breaking change`.

## Who reviews what

GitHub asks the code owners in [`.github/CODEOWNERS`](.github/CODEOWNERS) to review every pull request. For now that
is Carlos Uribe (@carluri) and James Fowler (@jdefowler) for every part of the code. Their reviews are requested, not
required.

| Area | Code | Reviewers |
|---|---|---|
| SPECT | `projectors/SPECT`, `io/SPECT`, `transforms/SPECT` | @carluri, @jdefowler |
| PET | `projectors/PET`, `io/PET`, `transforms/PET`, `utils/sss.py` | @carluri, @jdefowler |
| CT | `projectors/CT`, `io/CT` | @carluri, @jdefowler |
| Algorithms, priors, likelihoods | `algorithms`, `priors`, `likelihoods` | @carluri, @jdefowler |
| Docs, tutorials, CI | `docs`, `.github` | @carluri, @jdefowler |

## Security

Please don't report security problems in public issues; see [SECURITY.md](SECURITY.md).

Contributors are credited in the release notes, and substantial features earn co-authorship on the next
PyTomography paper.
