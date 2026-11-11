"""Layer 0: engineering-unit parsing, SI conversion, and engineering-notation formatting.

A unit string such as ``mV``, ``uA/MHz`` or ``kSPS`` is split into an SI prefix and a base
unit; compound units ``a/b`` are handled as a ratio.  ``to_si`` returns the value in SI base
units together with a canonical dimension string so that values with different prefixes can
be compared exactly (after floating-point rounding).
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

PREFIX = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "M": 1e6, "G": 1e9}
BASES = ("V", "A", "Hz", "ohm", "W", "s", "F", "H", "dB", "degC", "%", "ppm", "B", "bit", "SPS", "LSB")
_ALIASES = (("µ", "u"), ("μ", "u"), ("Ω", "ohm"), ("ohms", "ohm"), ("Ohm", "ohm"), ("°C", "degC"), ("℃", "degC"))


def canon(unit: str) -> str:
    """Normalise unicode variants (micro sign, ohm sign, degree-C) of a unit string."""
    s = unit.strip()
    for a, b in _ALIASES:
        s = s.replace(a, b)
    return s


def _simple(s: str) -> Tuple[float, str]:
    if s in BASES:
        return 1.0, s
    if len(s) > 1 and s[0] in PREFIX and s[1:] in BASES:
        return PREFIX[s[0]], s[1:]
    raise ValueError(f"unknown unit {s!r}")


def parse_unit(unit: str) -> Tuple[float, str]:
    """Return ``(scale_to_si, dimension)`` for a unit string; raises ValueError if unknown."""
    s = canon(unit)
    if "/" in s:
        a, b = s.split("/", 1)
        sa, da = _simple(a)
        sb, db = _simple(b)
        return sa / sb, f"{da}/{db}"
    return _simple(s)


def is_unit(unit: str) -> bool:
    try:
        parse_unit(unit)
        return True
    except ValueError:
        return False


def to_si(value: float, unit: str) -> Tuple[float, str]:
    scale, dim = parse_unit(unit)
    return value * scale, dim


def convert(value: float, src: str, dst: str) -> float:
    """Convert between two units of the same dimension."""
    v, d1 = to_si(value, src)
    s2, d2 = parse_unit(dst)
    if d1 != d2:
        raise ValueError(f"dimension mismatch {d1} vs {d2}")
    return v / s2


def choose_unit(value_si: float, candidates: Sequence[str]) -> str:
    """Pick the candidate unit that puts |value| in [1, 1000), else the closest to that window."""
    best = None
    for u in candidates:
        sc, _ = parse_unit(u)
        x = abs(value_si) / sc
        if 1.0 <= x < 1000.0:
            return u
        score = abs(math.log10(max(x, 1e-30)) - 1.5)
        if best is None or score < best[0]:
            best = (score, u)
    return best[1]


def round_sig(x: float, sig: int = 3) -> float:
    if x == 0:
        return 0.0
    return float(f"{x:.{sig - 1}e}")


def fmt_num(x: float, sig: int = 3) -> str:
    """Format a number with ``sig`` significant digits and no exponent."""
    s = f"{x:.{sig}g}"
    if "e" in s or "E" in s:
        s = f"{float(s):f}".rstrip("0").rstrip(".")
    return s


def format_eng(value_si: float, base: str, sig: int = 3) -> str:
    """Engineering notation, e.g. ``0.0033 V -> '3.3 mV'``."""
    if value_si == 0:
        return f"0 {base}"
    order = [("G", 1e9), ("M", 1e6), ("k", 1e3), ("", 1.0), ("m", 1e-3), ("u", 1e-6), ("n", 1e-9), ("p", 1e-12)]
    for p, sc in order:
        if abs(value_si) >= sc * (1 - 1e-12):
            return f"{fmt_num(value_si / sc, sig)} {p}{base}"
    return f"{fmt_num(value_si / 1e-12, sig)} p{base}"


def close(a: float, b: float, rel: float = 1e-3) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)
