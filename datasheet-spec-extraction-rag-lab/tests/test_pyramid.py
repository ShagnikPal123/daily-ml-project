"""Pyramid structure: each module imports only from strictly lower layers; the apex reaches every layer."""
import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "src" / "dsl"
LAYERS = {
    "schema": 0, "units": 0, "textnorm": 0, "viterbi": 0, "metrics": 0, "catalog": 0,
    "datasheetgen": 1,
    "chunker": 2, "rules": 2, "tagger": 2, "retrieval": 2,
    "extract": 3, "indexcompress": 3, "fixedpoint": 3, "confusion": 3,
    "qa": 4,
    "experiments": 5,
    "cli": 6,
}


def project_imports(mod: str):
    tree = ast.parse((PKG / f"{mod}.py").read_text())
    deps = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.level == 1:
            if n.module:
                deps.add(n.module.split(".")[0])
            else:
                deps.update(a.name for a in n.names)
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("dsl."):
                    deps.add(a.name.split(".")[1])
    return deps


def test_every_module_is_classified():
    mods = {p.stem for p in PKG.glob("*.py")} - {"__init__"}
    assert mods == set(LAYERS)


def test_imports_only_point_downward():
    for mod, layer in LAYERS.items():
        for dep in project_imports(mod):
            assert dep in LAYERS, (mod, dep)
            assert LAYERS[dep] < layer, f"{mod} (L{layer}) imports {dep} (L{LAYERS[dep]})"


def test_base_layer_has_no_project_dependencies():
    for mod, layer in LAYERS.items():
        if layer == 0:
            assert project_imports(mod) == set(), mod


def test_apex_depends_on_every_layer_and_module():
    seen, stack = set(), ["cli"]
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        stack.extend(project_imports(m))
    assert seen == set(LAYERS)
    assert {LAYERS[m] for m in seen} == set(range(max(LAYERS.values()) + 1))


def test_layer_table_is_contiguous():
    assert sorted(set(LAYERS.values())) == list(range(7))
