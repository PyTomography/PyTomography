# Contributing

PyTomography is built by the people who use it: physicists, engineers and students who needed a method that did not exist yet. If you have that need too, this section explains how to add it and get it merged.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Pick something to build
:link: feature-board
:link-type: doc

The feature board lists work we want done, with data to test on, where to start, and what "done" means.
:::

:::{grid-item-card} Run the tests
:link: testing
:link-type: doc

How the test suite is organised, and the contract tests every new projector and algorithm gets for free.
:::

:::{grid-item-card} Ask a question
:link: https://pytomography.discourse.group/

Not sure where to start, or whether an idea fits? Ask on Discourse before you write code.
:::
::::

## What you get for contributing

- **Credit.** Every contributor is named in the release notes and on the contributors page. Substantial features earn co-authorship on the next PyTomography paper.
- **A reviewer who replies.** Every issue and pull request gets a first reply within five working days. Each feature-board item has a named mentor.
- **Visibility.** Images made with your feature go into the [gallery](../gallery.md), credited with a link to your paper.

## Your first pull request

1. **Say what you plan to do.** Comment on the issue, or open one, so nobody duplicates your work and a maintainer can point you to the right place in the code.
2. **Fork and clone** the repository, then install it in editable mode with the development tools:

   ```bash
   git clone https://github.com/<your-username>/PyTomography.git
   cd PyTomography
   pip install -e ".[dev]"
   pre-commit install
   ```

3. **Create a branch from `main`.** `main` is the only long-lived branch. Releases are tags on it.
4. **Write a test that fails first.** For a bug, reproduce it in a test. For a new projector or algorithm, inherit the [contract tests](testing.md#contract-tests).
5. **Make the change** and run `pytest` until everything passes. A GPU is not needed; CI runs the GPU tests.
6. **Document it.** Public functions need a docstring that gives the shape, units and device of every tensor argument. A new feature needs a short tutorial notebook or an example in an existing one.
7. **Open a pull request against `main`.** The template asks what changed, how you tested it, and whether it changes numbers. CI runs the tests and builds a preview of the docs.

## Writing a tutorial

Tutorials are notebooks in `docs/source/notebooks`, stored with their outputs. To add one:

1. Save the notebook with its outputs. Make the last image output the one you want as the thumbnail, or set `thumbnail:` to the index of another.
2. Add an entry to `docs/source/tutorials/tutorials.yaml` with a title, modality, data source, topic and a one-sentence summary. The gallery, the filters, the sidebar and the Colab button are generated from it.
3. Publish any data the notebook needs on Zenodo, or use the existing tutorial records.

## Building these docs

```bash
pip install -e ".[doc]"
sphinx-autobuild docs/source docs/build/html --ignore "*/_generated/*"
```

Then open <http://127.0.0.1:8000>. Pages rebuild when you save a file.

```{toctree}
:hidden:

feature-board
testing
```
