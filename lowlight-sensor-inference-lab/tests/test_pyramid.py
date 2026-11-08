"""Structural tests: imports must point strictly downward and the apex must reach every layer."""
import ast
import pathlib

PKG = pathlib.Path(__file__).resolve().parents[1] / "lowlight"

LAYERS = {
    "fixedpoint": 0, "imgops": 0, "stats": 0, "data": 0,
    "sensor": 1, "classifiers": 1,
    "denoisers": 2, "ptc": 2,
    "pipeline": 3, "budget": 3,
    "experiments": 4, "report": 4,
    "cli": 5,
}
APEX = "cli"


def project_imports(mod: str) -> set[str]:
    tree = ast.parse((PKG / f"{mod}.py").read_text())
    found = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.level == 1:
            if n.module:
                found.add(n.module.split(".")[0])
            else:
                found.update(a.name for a in n.names)
        elif isinstance(n, ast.Import):
            found.update(a.name.split(".")[1] for a in n.names if a.name.startswith("lowlight."))
    return found


def closure(mod: str) -> set[str]:
    seen, stack = set(), [mod]
    while stack:
        for d in project_imports(stack.pop()):
            if d not in seen:
                seen.add(d)
                stack.append(d)
    return seen


def test_every_module_assigned_a_layer():
    mods = {p.stem for p in PKG.glob("*.py")} - {"__init__", "__main__"}
    assert mods == set(LAYERS)


def test_imports_point_strictly_downward():
    for mod, layer in LAYERS.items():
        for dep in project_imports(mod):
            assert dep in LAYERS, f"{mod} imports unknown module {dep}"
            assert LAYERS[dep] < layer, f"{mod} (L{layer}) imports {dep} (L{LAYERS[dep]})"


def test_layer0_has_no_project_dependencies():
    for mod, layer in LAYERS.items():
        if layer == 0:
            assert not project_imports(mod)


def test_apex_depends_on_every_layer_and_module():
    reach = closure(APEX)
    assert reach == set(LAYERS) - {APEX}
    assert {LAYERS[m] for m in reach} == set(range(5))


def test_no_import_cycles():
    for mod in LAYERS:
        assert mod not in closure(mod)


def test_every_module_has_a_test_file_exercising_it():
    tests = "\n".join(p.read_text() for p in (PKG.parent / "tests").glob("test_*.py") if p.name != "test_pyramid.py")
    for mod in LAYERS:
        assert f"lowlight import" in tests and mod in tests, mod
