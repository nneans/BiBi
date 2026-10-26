"""Keep heavy CPU work off this PC's performance cores.

2026-10-25: the same numba compile job failed 11 of 24 times pinned to the P-cores (logical CPUs 0-15:
access violations, stack overruns, impossible Python errors) and 0 of 24 times pinned to the E-cores
(logical CPUs 16-31). That is the known 13th/14th-gen Intel P-core degradation, not a code bug. Until the
CPU is fixed (BIOS microcode 0x12B+ or replacement), analysis processes pin themselves to the E-cores.
Set BIBI_ALL_CORES=1 to skip the pinning once the hardware is fixed.
"""
import ctypes
import os
import sys

E_CORE_MASK = 0xFFFF0000  # i9-14900KF: 8 P-cores x 2 threads = CPUs 0-15, 16 E-cores = CPUs 16-31


def pin_to_efficiency_cores():
    """Call before importing numba/torch so every thread they start inherits the mask. Returns True if pinned."""
    if sys.platform != 'win32' or os.environ.get('BIBI_ALL_CORES') == '1' or os.cpu_count() != 32:
        return False
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    ok = kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), E_CORE_MASK)
    if ok:
        # 16 E-cores: keep numba/BLAS pools inside them
        for name in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
            os.environ.setdefault(name, '8')
    use_own_numba_cache()
    return bool(ok)


def use_own_numba_cache():
    """The numba cache inside site-packages (pynndescent, umap) was found corrupted on 2026-10-25. Use a cache of
    our own instead; it must be a short path, because numba cache file names are long and Windows caps paths at
    260 characters (a cache under the Claude scratchpad failed with FileNotFoundError)."""
    path = os.path.join(os.path.expanduser('~'), '.bibi-numba')
    os.makedirs(path, exist_ok=True)
    os.environ.setdefault('NUMBA_CACHE_DIR', path)
