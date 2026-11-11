"""Layer 0: catalogue of component families and their datasheet parameters.

Value ranges are in SI base units and loosely follow what real parts of each family quote;
they are *synthetic* (no real part is modelled).  ``aliases`` are other names engineers use
for the same quantity, used to paraphrase questions.  ``cond`` describes a test condition
that scales the typical value as ``(x / x_ref) ** alpha`` for the first listed value x_ref.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

M, T, X = "min", "typ", "max"


@dataclass(frozen=True)
class ParamDef:
    pid: str
    names: Tuple[str, ...]
    symbol: str
    units: Tuple[str, ...]
    lo: float
    hi: float
    pats: Tuple[Tuple[str, ...], ...]
    aliases: Tuple[str, ...] = ()
    cond: Optional[Tuple[str, str, Tuple[float, ...], float]] = None  # (symbol, unit, values, alpha)
    choices: Tuple[float, ...] = ()


def P(pid, names, symbol, units, lo, hi, pats, aliases=(), cond=None, choices=()):
    return ParamDef(pid, tuple(names), symbol, tuple(units), lo, hi, tuple(tuple(p) for p in pats), tuple(aliases), cond, tuple(choices))


FAMILIES: Dict[str, Dict] = {
    "ldo": {
        "title": "Low-Dropout Linear Regulator",
        "prefixes": ("TLV", "LDK", "MIC"),
        "params": (
            P("vin", ["Input Voltage Range", "Input Supply Voltage"], "VIN", ["V"], 1.8, 16, [(M, X)], ["operating input voltage"]),
            P("dropout", ["Dropout Voltage"], "VDO", ["mV", "V"], 0.05, 0.6, [(T, X), (T,)], ["headroom voltage", "minimum input-output differential"],
              cond=("IOUT", "mA", (100.0, 300.0, 500.0, 1000.0), 0.8)),
            P("iq", ["Quiescent Current", "Ground Pin Current"], "IQ", ["nA", "uA", "mA"], 0.5e-6, 300e-6, [(T, X), (T,)], ["standby current", "idle current"]),
            P("psrr", ["Power Supply Rejection Ratio", "Ripple Rejection"], "PSRR", ["dB"], 40, 90, [(M, T), (T,)], ["supply ripple attenuation"],
              cond=("f", "kHz", (0.1, 1.0, 10.0, 100.0), -0.12)),
            P("iout", ["Maximum Output Current", "Output Current"], "IOUT", ["mA", "A"], 0.1, 3.0, [(M,), (M, T)], ["load current capability"]),
            P("tsd", ["Thermal Shutdown Temperature"], "TSD", ["degC"], 140, 170, [(T,)], ["overtemperature trip point"]),
            P("noise", ["Output Noise Voltage"], "eN", ["uV"], 10e-6, 200e-6, [(T,)], ["output noise"]),
        ),
    },
    "opamp": {
        "title": "Precision Operational Amplifier",
        "prefixes": ("OPA", "LMV", "AD8"),
        "params": (
            P("gbw", ["Gain Bandwidth Product", "Unity-Gain Bandwidth"], "GBW", ["kHz", "MHz"], 0.2e6, 500e6, [(T,), (M, T)], ["unity-gain frequency", "gain-bandwidth"]),
            P("slew", ["Slew Rate"], "SR", ["V/us"], 0.1e6, 500e6, [(T,)], ["output slew"]),
            P("vos", ["Input Offset Voltage"], "VOS", ["uV", "mV"], 20e-6, 5e-3, [(T, X)], ["offset voltage", "input offset"]),
            P("ib", ["Input Bias Current"], "IB", ["pA", "nA"], 1e-12, 500e-9, [(T, X)], ["bias current"]),
            P("icc", ["Supply Current", "Quiescent Current per Amplifier"], "ICC", ["uA", "mA"], 30e-6, 8e-3, [(T, X), (T,)], ["power consumption current"]),
            P("cmrr", ["Common-Mode Rejection Ratio"], "CMRR", ["dB"], 70, 130, [(M, T), (M,)], ["common mode rejection"]),
            P("vs", ["Supply Voltage Range", "Operating Supply Voltage"], "VS", ["V"], 1.8, 36, [(M, X)], ["power supply range"]),
        ),
    },
    "adc": {
        "title": "Analog-to-Digital Converter",
        "prefixes": ("ADS", "MCP", "LTC"),
        "params": (
            P("res", ["Resolution"], "N", ["bit"], 8, 24, [(T,)], ["number of bits"], choices=(8, 10, 12, 14, 16, 18, 20, 24)),
            P("fs", ["Sampling Rate", "Conversion Rate"], "fS", ["kSPS", "MSPS"], 10e3, 100e6, [(X,)], ["throughput", "sample frequency"]),
            P("inl", ["Integral Nonlinearity"], "INL", ["LSB"], 0.2, 4.0, [(T, X)], ["integral linearity error"]),
            P("dnl", ["Differential Nonlinearity"], "DNL", ["LSB"], 0.1, 2.0, [(T, X)], ["differential linearity error"]),
            P("snr", ["Signal-to-Noise Ratio"], "SNR", ["dB"], 45, 100, [(M, T), (T,)], ["signal noise ratio"],
              cond=("fIN", "kHz", (1.0, 10.0, 100.0), -0.05)),
            P("pd", ["Power Consumption", "Power Dissipation"], "PD", ["uW", "mW"], 0.2e-3, 0.5, [(T, X), (T,)], ["power draw"]),
        ),
    },
    "buck": {
        "title": "Synchronous Buck Converter",
        "prefixes": ("TPS", "LMR", "MPQ"),
        "params": (
            P("fsw", ["Switching Frequency"], "fSW", ["kHz", "MHz"], 100e3, 3e6, [(M, T, X), (T,)], ["oscillator frequency", "switching rate"]),
            P("eff", ["Peak Efficiency"], "EFF", ["%"], 80, 98, [(T,)], ["conversion efficiency"]),
            P("iout", ["Maximum Output Current"], "IOUT", ["A"], 0.5, 10, [(M,)], ["rated load current"]),
            P("rdson", ["High-Side MOSFET On-Resistance", "High-Side Switch Resistance"], "RDS(on)", ["mohm", "ohm"], 0.01, 0.3, [(T, X)], ["high-side resistance"]),
            P("iq", ["Quiescent Current"], "IQ", ["uA"], 5e-6, 60e-6, [(T, X)], ["standby current", "no-load current"]),
            P("tss", ["Soft-Start Time"], "tSS", ["us", "ms"], 0.2e-3, 8e-3, [(T,)], ["start-up ramp time"]),
        ),
    },
    "mcu": {
        "title": "Ultra-Low-Power Microcontroller",
        "prefixes": ("STM", "MSP", "PIC"),
        "params": (
            P("fclk", ["Maximum CPU Frequency"], "fCPU", ["MHz"], 8e6, 480e6, [(X,)], ["core clock speed"]),
            P("flash", ["Flash Memory"], "FLASH", ["kB", "MB"], 16e3, 2048e3, [(T,)], ["program memory"], choices=(16e3, 32e3, 64e3, 128e3, 256e3, 512e3, 1024e3, 2048e3)),
            P("ram", ["SRAM"], "RAM", ["kB"], 4e3, 512e3, [(T,)], ["data memory"], choices=(4e3, 8e3, 16e3, 32e3, 64e3, 128e3, 256e3, 512e3)),
            P("irun", ["Active Mode Current"], "IRUN", ["uA/MHz"], 20e-12, 300e-12, [(T,)], ["run mode current"]),
            P("istop", ["Stop Mode Current"], "ISTOP", ["nA", "uA"], 0.2e-6, 20e-6, [(T, X)], ["deep sleep current", "standby current"]),
            P("vdd", ["Supply Voltage"], "VDD", ["V"], 1.62, 3.6, [(M, X)], ["operating voltage"]),
        ),
    },
}

# Absolute-maximum-rating rows (generated for every part): (pid, name, symbol, unit, lo, hi, which)
ABS_ROWS = (
    ("abs_vs", "Absolute Maximum Supply Voltage", "VMAX", "V", 6.0, 40.0, "max"),
    ("abs_tstg", "Storage Temperature", "TSTG", "degC", None, None, "minmax"),
    ("abs_tj", "Junction Temperature", "TJ", "degC", 125.0, 150.0, "max"),
)

DESCRIPTIONS = {
    "ldo": "The device is a low-noise linear regulator for battery and portable systems.",
    "opamp": "The amplifier targets sensor interfaces and precision signal conditioning.",
    "adc": "The converter targets data acquisition and industrial measurement.",
    "buck": "The converter targets point-of-load regulation in compact designs.",
    "mcu": "The controller targets edge sensing nodes with long battery life.",
}


def all_params():
    for fam in FAMILIES.values():
        yield from fam["params"]
