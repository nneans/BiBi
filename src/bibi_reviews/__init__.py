"""Local review analysis experiments.

Importing the package pins the process to this PC's E-cores (see cpu.py): a script that skipped the explicit call
crashed with an impossible TypeError on 2026-10-26. The process mask also covers threads started before it.
"""
from .cpu import pin_to_efficiency_cores

pin_to_efficiency_cores()
