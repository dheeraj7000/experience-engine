"""Benchmark adapters for external evaluation suites.

Each adapter bridges an upstream benchmark into the Experience Engine's
TaskFamily/Environment protocol, enabling A0/A1/A2 comparison on established
evaluation sets.
"""
from .lifelongagentbench import LifelongAgentBenchFamily

__all__ = ["LifelongAgentBenchFamily"]
