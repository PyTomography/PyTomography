# Using these docs with AI assistants

Coding assistants such as Claude often answer from what they saw during training, which may be an older version of PyTomography. These docs publish machine-readable versions of every page so an assistant can work from the current release instead.

## What is available

| File | What it contains |
|---|---|
| <a href="llms.txt"><code>llms.txt</code></a> | A short index of the documentation with a link to every page. |
| <a href="llms-full.txt"><code>llms-full.txt</code></a> | The full text of every guide, tutorial and top-level API page in one file. |
| `_md/<page>.md` | A Markdown version of a single page. Tutorials are converted from notebooks, with figures left out. Every page has a **Copy page as Markdown** button that copies it for you. |

## Giving the docs to an assistant

- **In a chat:** press **Copy page as Markdown** on the page you are working from and paste it in with your question.
- **In an agent such as Claude Code:** point it at `llms.txt` and ask it to read the pages it needs, for example: *"Read https://pytomography.readthedocs.io/en/latest/llms.txt, then help me reconstruct my Lu-177 DICOM data."*
- **For a whole project:** save `llms-full.txt` next to your code so the agent can search it.

## Tips that avoid common mistakes

- Tell the assistant which PyTomography version you have installed (`pytomography.__version__`). v4 changed the PET requirements and the PSF interface.
- Ask it to check tensor shapes and devices against the docstrings. Projections, objects and attenuation maps each have a fixed axis order, and mixing CPU and GPU tensors is the most common error.
- Ask for code that runs on a small subset of angles first, then scale up.
