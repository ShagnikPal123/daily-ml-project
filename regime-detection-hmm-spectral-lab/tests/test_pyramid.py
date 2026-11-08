"""Dependency-direction and coverage checks for the layered architecture."""
import ast
import pathlib

PKG = pathlib.Path(__file__).resolve().parents[1] / "regimelab"
LAYERS = {"config": 0, "synth": 0, "metrics": 0, "filters": 1, "hmm": 1, "features": 2, "strategy": 2,
          "detectors": 3, "evaluation": 4, "experiments": 5, "cli": 6}


def imports(mod):
    tree = ast.parse((PKG / f"{mod}.py").read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            if node.module:
                found.add(node.module.split(".")[0])
            else:
                found.update(a.name for a in node.names)
    return found & set(LAYERS)


def test_every_module_is_assigned_a_layer():
    on_disk = {p.stem for p in PKG.glob("*.py")} - {"__init__", "__main__"}
    assert on_disk == set(LAYERS)


def test_imports_only_point_downward():
    for mod, layer in LAYERS.items():
        for dep in imports(mod):
            assert LAYERS[dep] < layer, f"{mod} (L{layer}) imports {dep} (L{LAYERS[dep]})"


def test_apex_depends_on_every_layer():
    seen, stack = set(), ["cli"]
    while stack:
        m = stack.pop()
        for d in imports(m):
            if d not in seen:
                seen.add(d)
                stack.append(d)
    assert seen == set(LAYERS) - {"cli"}
    assert {LAYERS[m] for m in seen} == set(range(6))
