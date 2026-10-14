"""Run a command without letting it take the machine's memory.

    python .github/run_capped.py --max-gb 24 --min-free-gb 16 -- python -m pytest -ra

The command and everything it starts are stopped if they use more than --max-gb, or if the machine's free memory
falls below --min-free-gb (free RAM and, on Windows, uncommitted memory). On Windows they also run in a job object
whose committed memory is capped at --max-gb, so an allocation past the cap fails inside the command instead of
exhausting the machine. Used by the GPU workflow, whose runner may be someone's workstation.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading

import psutil


def free_gb() -> float:
    free = psutil.virtual_memory().available
    if os.name == "nt":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong) for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                                                  "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
        status = MEMORYSTATUSEX(dwLength=ctypes.sizeof(MEMORYSTATUSEX))
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            free = min(free, status.ullAvailPageFile)
    return free / 1e9


def cap_this_process(limit_gb: float) -> bool:
    """Windows only: put this process in a job whose committed memory is capped; its children inherit the job."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", ctypes.c_ulonglong * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    job = k32.CreateJobObjectW(None, None)
    info = EXTENDED()
    info.BasicLimitInformation.LimitFlags = 0x200 | 0x2000  # JOB_OBJECT_LIMIT_JOB_MEMORY | KILL_ON_JOB_CLOSE
    info.JobMemoryLimit = int(limit_gb * 1e9)
    return bool(job and k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
                and k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-gb", type=float, required=True, help="most memory the command may use")
    ap.add_argument("--min-free-gb", type=float, default=16, help="stop the command if less than this is free")
    ap.add_argument("command", nargs=argparse.REMAINDER, help="the command, after --")
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if free_gb() < args.min_free_gb:
        print(f"run_capped: only {free_gb():.1f} GB free, need {args.min_free_gb:.0f} GB; not starting", flush=True)
        return 1
    if os.name == "nt" and not cap_this_process(args.max_gb + 0.5):  # + this script itself
        print("run_capped: could not set the hard cap; the watchdog still applies", flush=True)

    proc = subprocess.Popen(command)
    tripped: list[str] = []

    def watch() -> None:
        me = psutil.Process()
        while proc.poll() is None:
            try:
                kids = me.children(recursive=True)
                used = sum((getattr(m, "private", None) or m.rss) / 1e9 for m in (k.memory_info() for k in kids))
            except psutil.Error:
                continue
            free = free_gb()
            if used > args.max_gb or free < args.min_free_gb:
                tripped.append(f"used {used:.1f} GB with {free:.1f} GB free")
                for k in kids:
                    try:
                        k.kill()
                    except psutil.Error:
                        pass
                return
            threading.Event().wait(1.0)

    threading.Thread(target=watch, daemon=True).start()
    code = proc.wait()
    if tripped:
        print(f"run_capped: stopped to protect the machine's memory ({tripped[0]}; limits {args.max_gb:g} GB used, "
              f"{args.min_free_gb:g} GB free)", flush=True)
        return 1
    return code


if __name__ == "__main__":
    sys.exit(main())
