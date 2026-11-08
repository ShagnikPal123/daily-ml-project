"""lowlight - sensor-to-inference lab for photon-starved edge vision.

Layers (imports only point downward):
  0 fixedpoint, imgops, stats, data   1 sensor, classifiers   2 denoisers, ptc
  3 pipeline, budget                  4 experiments, report   5 cli (apex)
"""
__version__ = "1.0.0"
LAYERS = {
    0: ("fixedpoint", "imgops", "stats", "data"),
    1: ("sensor", "classifiers"),
    2: ("denoisers", "ptc"),
    3: ("pipeline", "budget"),
    4: ("experiments", "report"),
    5: ("cli",),
}
