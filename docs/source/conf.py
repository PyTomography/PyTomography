# Sphinx configuration for the PyTomography documentation.
# Build locally with:  sphinx-autobuild docs/source docs/build/html
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath("_ext"))

# -- Project information -----------------------------------------------------

project = "PyTomography"
author = "Luke Polson and the PyTomography contributors"
copyright = "2023-2026, the PyTomography contributors"

# The site documents the upcoming 4.0 release; the release workflow will set this
# from the git tag once pyproject.toml is bumped.
release = os.environ.get("PYTOMOGRAPHY_DOCS_VERSION", "4.0 preview")
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",                      # Markdown and notebooks (includes myst_parser)
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx_copybutton",
    "autoapi.extension",
    "pytomo_docs",                  # _ext/pytomo_docs.py: gallery, launch bars, llms.txt
    "pytomo_viewer",                # _ext/pytomo_viewer.py: the 3D image viewer on tutorial pages and cards
]

source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}

exclude_patterns = [
    "_build", "_generated", "**.ipynb_checkpoints",
    "index2.md",
    # Notebooks not yet listed in tutorials/tutorials.yaml
    "notebooks/t_dicom_algorithms.ipynb",
]

# Notebooks are rendered with their stored outputs; CI executes them separately.
nb_execution_mode = "off"
nb_merge_streams = True
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist", "html_image", "attrs_inline"]
myst_heading_anchors = 3

# -- API reference -----------------------------------------------------------

autoapi_dirs = ["../../src/pytomography"]
autoapi_ignore = ["*/tests/*"]
autoapi_root = "api"
autoapi_add_toctree_entry = False
# Keep the generated pages: deleting them at the end of a build races with live-reload rebuilds
autoapi_keep_files = True
autoapi_options = ["members", "undoc-members", "show-inheritance", "show-module-summary"]
autodoc_typehints = "description"
suppress_warnings = ["autoapi.python_import_resolution", "myst.header", "misc.highlighting_failure", "mystnb.unknown_mime_type"]


def _skip_attributes(app, what, name, obj, skip, options):
    return True if what == "attribute" else skip


def setup(app):
    app.connect("autoapi-skip-member", _skip_attributes)


# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "PyTomography"
html_logo = "images/PT1.png"
html_favicon = "images/PT1.png"
html_static_path = ["_static", "_generated"]
html_css_files = [
    "https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap",
    "css/pytomo.css",
]
html_js_files = ["js/pytomo.js"]
html_sidebars = {"index": [], "install": [], "migration": [], "ai": []}
html_context = {
    "github_user": "PyTomography",
    "github_repo": "PyTomography",
    "github_version": "main",
    "doc_path": "docs/source",
    "default_mode": "auto",
}
html_theme_options = {
    "logo": {"text": "PyTomography"},
    "navbar_align": "left",
    # Installation, Tutorials, Gallery, API and Contribute in the top bar; Migrate to v4, Concepts and the rest under More
    "header_links_before_dropdown": 5,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {"name": "GitHub", "url": "https://github.com/PyTomography/PyTomography", "icon": "fa-brands fa-github"},
        {"name": "Discourse", "url": "https://pytomography.discourse.group/", "icon": "fa-solid fa-comments"},
        {"name": "PyPI", "url": "https://pypi.org/project/pytomography/", "icon": "fa-brands fa-python"},
    ],
    "announcement": "You are reading the preview of the PyTomography 4.0 documentation. Release target: 30 October 2026.",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    # tutorial pages use the full width: their headings sit in the section navigation instead (js/pytomo-tutorial.js).
    # Each is named exactly, since a page matching two wildcard patterns draws a warning.
    "secondary_sidebar_items": {"**": ["page-toc", "edit-this-page"], **{
        "notebooks/" + os.path.splitext(n)[0]: [] for n in os.listdir(os.path.join(os.path.dirname(__file__), "notebooks"))
        if n.endswith(".ipynb")}},
    "footer_start": ["copyright"],
    "footer_end": [],
    "pygments_light_style": "friendly",
    "pygments_dark_style": "github-dark",
}
