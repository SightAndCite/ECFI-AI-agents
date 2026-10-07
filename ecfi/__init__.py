"""ECFI: Epistemic-Compute Feedback Instability testbed.

Modules
-------
linear     local cross-layer model, Jury conditions, Theorem 1
policies   uncertainty-proportional and congestion-aware verification laws
schedules  piecewise-constant exogenous excitation
sim        SimPy discrete-event testbed with switchable epistemic coupling
identify   pulse-based system identification with bootstrap intervals
detect     transition detection and early-warning signals
ela        epistemic load amplification
"""

from .linear import (  # noqa: F401
    LocalParams,
    A_matrix,
    jury,
    is_schur,
    spectral_radius,
    kappa_crit,
    kappa_delta,
    exactness_holds,
    paradox_holds,
    u_star,
    stability_margin,
)
from .policies import make_policy  # noqa: F401
from .schedules import Schedule, constant, pulse_train, dither  # noqa: F401
from .sim import SimConfig, WindowSeries, run_sim, WAIT, RESOURCE_COUPLED  # noqa: F401

__version__ = "0.1.0"
