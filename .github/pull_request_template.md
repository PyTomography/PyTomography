## What this changes

<!-- One or two sentences. Link the issue it closes, e.g. "Closes #123". -->

## Why

## How it was tested

<!-- Tests added or changed, and their output. For a new projector or algorithm, the contract tests it inherits. -->

## Does it change reconstructed numbers?

<!-- "No", or what changes and by how much, with the evidence. -->

## Checklist

- [ ] Tests pass locally (`pytest`)
- [ ] New code has tests; a bug fix includes a test that failed before the fix
- [ ] Public functions have docstrings with tensor shapes, units and devices
- [ ] If a tutorial notebook changed, `python docs/tools/export_scripts.py` was run
