"""Fill README.template.md with tables from results/tables.md and measured counts (run after `dsl run`)."""
import json
import re
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
R = json.loads((root / "results/results.json").read_text())
cfg = R["config"]
pieces = [p.strip() for p in re.split(r"(?m)^(?=### T\d)", (root / "results/tables.md").read_text()) if p.strip()]
t = (root / "README.template.md").read_text()
for i, p in enumerate(pieces):
    t = t.replace("{{P%d}}" % i, re.sub(r"^### ", "**", p.split("\n", 1)[0]) + "**\n\n" + p.split("\n", 1)[1].strip())
valid = subprocess.run([sys.executable, str(root / "tests/test_reference.py")], capture_output=True, text=True, check=True).stdout.strip()
nonblank = lambda g: sum(1 for f in g for ln in f.read_text().splitlines() if ln.strip())
ntests = len(re.findall(r"^def test_", "".join(f.read_text() for f in (root / "tests").glob("test_*.py")), re.M))
rep = {"VALID": valid, "RUNTIME": "%.0f" % R["runtime_s"], "NTEST": cfg["n_test"], "NTRAIN": cfg["n_train"], "NQ": cfg["n_questions"],
       "NSEEDS": len(cfg["seeds"]), "LOC_SRC": nonblank((root / "src/dsl").glob("*.py")), "LOC_TESTS": nonblank((root / "tests").glob("*.py")), "NTESTS": ntests}
for k, v in rep.items():
    t = t.replace("{{%s}}" % k, str(v))
assert "{{" not in t, re.findall(r"\{\{\w+\}\}", t)
(root / "README.md").write_text(t)
print(rep["LOC_SRC"], rep["LOC_TESTS"], rep["NTESTS"], len(t.splitlines()))
