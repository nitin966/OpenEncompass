"""Compare vanilla vs encompass."""

from pathlib import Path


def lines(p):
    return len(
        [
            line
            for line in p.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    )


def valid(d):
    if not d.exists():
        return 0, 0
    files = list(d.rglob("*.py"))
    v = 0
    for f in files:
        try:
            compile(f.read_text(), f.name, "exec")
            v += 1
        except Exception:
            pass
    return len(files), v


base = Path("examples/code_translation/simple_experiment")
vl, el = lines(base / "baseline_vanilla.py"), lines(base / "encompass_version.py")
vf, vv = valid(base / "output/vanilla")
ef, ev = valid(base / "output/encompass")
print(f"Vanilla:   {vl} lines, {vv}/{vf} valid")
print(f"EnCompass: {el} lines, {ev}/{ef} valid")
print(f"Reduction: {(vl - el) * 100 // vl}%")

vf, vv = valid(base / "output/vanilla")
ef, ev = valid(base / "output/encompass")
print(f"Vanilla:   {vl} lines, {vv}/{vf} valid")
print(f"EnCompass: {el} lines, {ev}/{ef} valid")
print(f"Reduction: {(vl - el) * 100 // vl}%")
