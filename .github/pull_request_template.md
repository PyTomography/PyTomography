Closes #

<!-- The issue this pull request closes (required: a check fails without it). Open one first if there isn't one.
     For a trivial change, a maintainer can waive this with the "no-issue" label. -->

## What this changes

<!-- One or two sentences. -->

## Why

## How it was tested

<!-- Tests added or changed, and their output. For a new projector or algorithm, the contract tests it inherits. -->

## Does it change reconstructed numbers?

<!-- "No", or what changes and by how much, with the evidence. -->

## Checklist

- [ ] The description closes an issue
- [ ] Tests pass locally (`pytest`)
- [ ] New code has tests; a bug fix includes a test that failed before the fix
- [ ] Public functions have docstrings with tensor shapes, units and devices
- [ ] If a tutorial notebook changed, `python docs/tools/export_scripts.py` was run
