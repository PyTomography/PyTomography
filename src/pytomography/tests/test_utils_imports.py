"""The modules of pytomography.utils import from their sibling modules, not from the package: pytomography/utils/
__init__.py imports them, so a module importing a name from the package is an import cycle (utils/scatter.py's
"from pytomography.utils import get_1d_gaussian_kernel" was the docs build's only warning, sphinx-autoapi's
"Unknown type: placeholder"). Importing a submodule from the package is fine."""
from __future__ import annotations

import ast
from pathlib import Path

import pytomography.utils

UTILS = Path(pytomography.utils.__file__).parent


def test_utils_modules_do_not_import_names_from_their_package():
    submodules = {p.stem for p in UTILS.glob('*.py')} | {p.name for p in UTILS.iterdir() if p.is_dir()}
    offending = []
    for path in sorted(UTILS.glob('*.py')):
        if path.name == '__init__.py':
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding='utf8'))):
            if not isinstance(node, ast.ImportFrom):
                continue
            from_package = (node.level == 1 and node.module is None) or (node.level == 0 and node.module == 'pytomography.utils')
            if from_package:
                offending += [f'{path.name}: {alias.name}' for alias in node.names if alias.name not in submodules]
    assert offending == []


def test_scatter_uses_the_kernel_of_misc():
    from pytomography.utils import misc, scatter
    assert scatter.get_1d_gaussian_kernel is misc.get_1d_gaussian_kernel
