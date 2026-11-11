"""Dependency-direction and coverage checks for the layer pyramid."""
import ast
import pathlib

PKG = pathlib.Path(__file__).resolve().parents[1] / "motorctl"
LAYERS = {
    "plant": 0, "fixedpoint": 0, "metrics": 0, "riccati": 0,
    "controllers": 1, "estimators": 1,
    "fixedpid": 2, "lqg": 2,
    "simulate": 3, "factory": 3,
    "experiments": 4,
    "cli": 5,
}


def _imports(name):
    tree = ast.parse((PKG / f"{name}.py").read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "motorctl":
                found.update(a.name for a in node.names)
            elif node.module.startswith("motorctl."):
                found.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("motorctl."):
                    found.add(a.name.split(".")[1])
    return found


def test_layer_map_covers_every_module():
    mods = {p.stem for p in PKG.glob("*.py") if p.stem != "__init__"}
    assert mods == set(LAYERS)


def test_imports_only_point_downward():
    for mod, layer in LAYERS.items():
        for dep in _imports(mod):
            assert dep in LAYERS, f"{mod} imports unknown {dep}"
            assert LAYERS[dep] < layer, f"{mod} (L{layer}) imports {dep} (L{LAYERS[dep]})"


def test_layer0_has_no_project_dependencies_except_within_layer0():
    for mod, layer in LAYERS.items():
        if layer == 0:
            assert all(LAYERS[d] == 0 for d in _imports(mod))


def test_apex_depends_on_every_module_transitively():
    seen, stack = set(), ["cli"]
    while stack:
        m = stack.pop()
        for d in _imports(m):
            if d not in seen:
                seen.add(d)
                stack.append(d)
    assert seen == set(LAYERS) - {"cli"}
