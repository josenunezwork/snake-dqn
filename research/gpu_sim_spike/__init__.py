"""Experimental device-resident dense-search world state.

This package is deliberately isolated from the governed T34 simulators.  It is
for the GPU simulator spike only and is never a serving or gate authority.
"""

from .device_sim import DeviceSim

__all__ = ["DeviceSim"]
