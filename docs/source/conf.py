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


_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")


def _skip_member(app, what, name, obj, skip, options):
    """Leave out class attributes, show ``__call__``, and show what a package imports from its private modules.

    Calling an algorithm (``OSEM(likelihood)(n_iters=4, n_subsets=8)``) is how it is used, so a documented
    ``__call__`` is shown, although other special methods are not. ``pytomography.io`` defines ``load_image`` in
    ``io/_load.py``: the private module has no page, so without this the function would have none either. It is
    documented as ``pytomography.io.load_image``, where users import it from.
    """
    if what == "attribute":
        return True
    if what == "method" and obj.short_name == "__call__" and obj.docstring and not obj.obj.get("hide"):
        return False
    original = obj.obj.get("original_path", "")
    if skip and original and not obj.obj.get("hide") and not obj.short_name.startswith("_"):
        defined_in, importer = original.rsplit(".", 1)[0], name.rsplit(".", 1)[0]
        is_package = os.path.isfile(os.path.join(_SRC, *importer.split("."), "__init__.py"))
        if is_package and any(part.startswith("_") for part in defined_in.split(".")):
            return False
    return skip


def _link_imported_names(app, env, node, contnode):
    """Link a name as users import it to the page that documents it.

    ``pytomography.metadata.ObjectMeta`` (in a type hint or a docstring) is documented as
    ``pytomography.metadata.metadata.ObjectMeta``, where it is defined: follow the package's import to it. Members
    (``pytomography.projectors.SystemMatrix.forward``) follow their class.
    """
    if node.get("refdomain") != "py":
        return None
    objects = getattr(env, "autoapi_all_objects", {})
    target, module, cls = node["reftarget"], node.get("py:module"), node.get("py:class")
    candidates = [target]
    if module:
        candidates += [f"{module}.{cls}.{target}"] if cls else []
        candidates += [f"{module}.{target}"]
    for candidate in candidates:
        parts = candidate.split(".")
        for _ in range(5):  # follow re-exports of re-exports
            # the longest prefix that was imported (a member of an imported class is not marked itself)
            for i in range(len(parts), 0, -1):
                obj = objects.get(".".join(parts[:i]))
                if obj is not None and obj.obj.get("original_path"):
                    break
            else:
                break
            parts = obj.obj["original_path"].split(".") + parts[i:]
        resolved = ".".join(parts)
        if resolved != candidate:
            ref = env.get_domain("py").resolve_xref(env, node["refdoc"], app.builder, node["reftype"], resolved,
                                                     node, contnode)
            if ref is not None:
                return ref
    return None


def setup(app):
    app.connect("autoapi-skip-member", _skip_member)
    app.connect("missing-reference", _link_imported_names)


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
html_sidebars = {"index": [], "install": [], "migration": [], "ai": [], "architecture": []}
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
    # Installation, Tutorials, Gallery, Architecture, API and Contribute in the top bar; Migrate to v4, Concepts and the
    # rest under More
    "header_links_before_dropdown": 6,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {"name": "GitHub", "url": "https://github.com/PyTomography/PyTomography", "icon": "fa-brands fa-github"},
        {"name": "Questions (GitHub Discussions)", "url": "https://github.com/PyTomography/PyTomography/discussions",
         "icon": "fa-solid fa-comments"},
        {"name": "Discourse (older answers)", "url": "https://pytomography.discourse.group/",
         "icon": "fa-brands fa-discourse"},
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
