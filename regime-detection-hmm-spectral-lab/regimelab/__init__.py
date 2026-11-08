"""regimelab: regime detection on return series viewed as a signal-processing problem.

Pyramid layers (each imports only from strictly lower layers):
  L0 config, synth, metrics   L1 filters, hmm   L2 features, strategy
  L3 detectors   L4 evaluation   L5 experiments   L6 cli (apex)
"""
__version__ = "0.1.0"
