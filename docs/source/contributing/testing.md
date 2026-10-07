# Testing

The test suite does two jobs: it stops changes from breaking reconstructions, and it tells you exactly what new code has to satisfy. Tests run in four tiers, so a pull request gets feedback in minutes while slower checks still run regularly.

| Tier | What it checks | When it runs |
|---|---|---|
| 1. Unit and contract | Adjointness, subsets, count conservation, input and output round trips, on tiny synthetic systems | Every pull request, on Linux, Windows and macOS, Python 3.10 to 3.13, CPU only. Under five minutes. |
| 2. Regression | Small reconstructions of cached tutorial data against stored reference images, and one test per fixed bug | Every pull request |
| 3. GPU and speed | CUDA code paths, the tests that projection loops never synchronise with the device, and benchmarks tracked over time | Nightly, and on pull requests labelled `run-gpu` |
| 4. Notebooks and docs | Every tutorial runs from start to finish; docs build without warnings; links resolve | Weekly, and before every release |

## Running the tests

```bash
pip install -e ".[dev]"
pytest                      # tiers 1 and 2
pytest -m gpu               # tier 3, needs a CUDA GPU
pytest -n auto              # in parallel
```

parallelproj is not needed for tiers 1 and 2: PET tests replace it with a small stand-in, as in `test_pet_sinogram_memory.py`.

(contract-tests)=
## Contract tests

Every system matrix in PyTomography must satisfy the same properties, whatever its geometry. Rather than each author writing these checks again, a new class inherits them:

```python
# src/pytomography/tests/contracts.py
class SystemMatrixContract:
    """Inherit this and implement make(); the checks below run automatically."""

    def make(self):
        """Return (system_matrix, object, projections) for a tiny problem."""
        raise NotImplementedError

    def test_forward_and_backward_are_adjoint(self):
        H, f, g = self.make()
        lhs = (H.forward(f) * g).sum()
        rhs = (f * H.backward(g)).sum()
        assert torch.isclose(lhs, rhs, rtol=1e-4)

    def test_subsets_partition_the_projections(self): ...
    def test_runs_on_cpu_and_cuda(self): ...
```

A contributor adding, for example, a pinhole projector writes only this:

```python
class TestPinholeSystemMatrix(SystemMatrixContract):
    def make(self):
        return tiny_pinhole_system()
```

`AlgorithmContract` does the same for reconstruction algorithms: it checks that the objective improves, that EM-type updates stay non-negative, and that subsets and full iterations agree on a small problem.

## Writing good tests

- **Make them small.** A 16³ object and a dozen angles catch most bugs and run in milliseconds.
- **Compare with something independent.** An analytic answer, an explicit matrix, or the previous implementation kept in the test, as `test_starguide_projector.py` does.
- **State tolerances.** Projectors accumulate with atomic additions on the GPU, so results vary at the 1e-7 level between runs. Assert with a tolerance and say why it is that size.
- **Reproduce the bug first.** A bug fix starts with a test that fails on `main`.
