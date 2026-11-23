#!/usr/bin/env python
"""
This module implements the class EnergyMeter, to measure the energy consumption of Python
functions or code chunks, segregating their energy usage per component (CPU, DRAM, GPU and
Hard Disk).

Platform support
----------------
- Linux + Intel/AMD: CPU and DRAM are measured with RAPL (read from the powercap sysfs files,
  sampled periodically so counter wraparounds are handled); disk via bpftrace (per process)
  or /proc/diskstats (whole system).
- macOS + Apple Silicon: CPU, GPU and ANE power are measured with Apple's `powermetrics` tool
  (requires sudo); DRAM with the kernel's IOReport "DRAM" energy counter (no sudo); disk via the kernel's per-process I/O counters
  (proc_pid_rusage) or IOKit's per-disk counters (whole system); no sudo, works with SIP.
- NVIDIA GPUs: measured with PyNVML. Multiple GPUs are supported and ALL detected GPUs are
  tracked by default. If there are no NVIDIA GPUs (or pynvml is absent), GPU tracking is simply
  skipped and reports 0 - it never raises.
"""
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

import subprocess
import os
import shlex
import json
import time
import threading
import platform
import re
import ctypes
import ctypes.util

# --- Defensive optional imports -------------------------------------------------------------
# pynvml (NVIDIA only) is not installable/usable on every
# platform (e.g. Apple Silicon Macs). We import it defensively so this module can still be
# imported and used for the measurements that ARE available on the host.
try:
    from pynvml import *  # noqa: F401,F403
    _PYNVML_AVAILABLE = True
except Exception:  # pragma: no cover - depends on host
    _PYNVML_AVAILABLE = False

    # Provide a fallback so `except NVMLError` clauses below remain valid even when pynvml
    # is not installed.
    class NVMLError(Exception):
        pass


def _is_apple_silicon():
    """Return True if running on an Apple Silicon (arm64) macOS machine."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _is_darwin():
    """Return True if running on macOS."""
    return platform.system() == "Darwin"


class _RusageInfoV2(ctypes.Structure):
    """Mirror of `struct rusage_info_v2` from <sys/resource.h> (macOS)."""
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
            "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
            "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
            "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
            "ri_child_pageins", "ri_child_elapsed_abstime",
            "ri_diskio_bytesread", "ri_diskio_byteswritten",
        )
    ]


_RUSAGE_INFO_V2 = 2
_libproc = None


def _darwin_disk_io_bytes(pid=None):
    """Return (bytes_read, bytes_written) that actually hit storage for `pid` (default: this
    process) since it started, using the kernel's per-process counters via proc_pid_rusage.

    Unlike DTrace, this needs no root and works with System Integrity Protection enabled.
    Note it counts physical disk I/O, so reads served from the page cache are not included.
    """
    global _libproc
    if _libproc is None:
        _libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib",
                               use_errno=True)
        _libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int,
                                             ctypes.POINTER(_RusageInfoV2)]
        _libproc.proc_pid_rusage.restype = ctypes.c_int
    info = _RusageInfoV2()
    rc = _libproc.proc_pid_rusage(os.getpid() if pid is None else pid, _RUSAGE_INFO_V2,
                                  ctypes.byref(info))
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"proc_pid_rusage failed: {os.strerror(err)}")
    return info.ri_diskio_bytesread, info.ri_diskio_byteswritten


# Rough figures for the internal SSD of Apple Silicon Macs (Apple publishes no power specs):
# ~3 GB/s sustained, ~3 W while transferring, ~0.05 W idle in its low-power state. That is
# about 1 J per GB moved. Pass your own values to EnergyMeter for better estimates.
APPLE_SSD_DEFAULTS = {"disk_avg_speed": 3e9, "disk_active_power": 3.0, "disk_idle_power": 0.05}

# Rough figures per disk type, used for any of the three disk values that are not given when
# the whole disk is tracked (disk_scope="system"). Typical spec-sheet values; real drives vary,
# so pass your own (see "How to get storage details" in the README) for better estimates.
DISK_DEFAULTS = {
    "apple_ssd": APPLE_SSD_DEFAULTS,
    "nvme": {"disk_avg_speed": 2e9, "disk_active_power": 5.0, "disk_idle_power": 0.5},
    "ssd": {"disk_avg_speed": 5e8, "disk_active_power": 3.0, "disk_idle_power": 0.3},
    "hdd": {"disk_avg_speed": 1.5e8, "disk_active_power": 6.5, "disk_idle_power": 4.5},
}


class _DarwinSystemDiskCounters:
    """System-wide bytes read/written by the Mac's internal disks (all processes), from the
    "Statistics" of each IOBlockStorageDriver in IOKit: the same counters `iostat` shows. No
    sudo needed. Disk images and external drives are skipped: a disk image's I/O already lands
    on the internal disk, and external drives have their own power figures."""

    _UTF8 = 0x08000100           # kCFStringEncodingUTF8
    _SINT64 = 4                  # kCFNumberSInt64Type
    _SEARCH_PARENTS = 0x1 | 0x2  # kIORegistryIterateRecursively | kIORegistryIterateParents

    def __init__(self):
        vp, u32 = ctypes.c_void_p, ctypes.c_uint32
        cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
        iokit = ctypes.CDLL(ctypes.util.find_library("IOKit"))
        cf.CFStringCreateWithCString.restype = vp
        cf.CFStringCreateWithCString.argtypes = [vp, ctypes.c_char_p, u32]
        cf.CFStringGetCString.restype = ctypes.c_bool
        cf.CFStringGetCString.argtypes = [vp, ctypes.c_char_p, ctypes.c_long, u32]
        cf.CFDictionaryGetValue.restype = vp
        cf.CFDictionaryGetValue.argtypes = [vp, vp]
        cf.CFNumberGetValue.restype = ctypes.c_bool
        cf.CFNumberGetValue.argtypes = [vp, ctypes.c_int, vp]
        cf.CFRelease.argtypes = [vp]
        iokit.IOServiceMatching.restype = vp
        iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
        iokit.IOServiceGetMatchingServices.restype = ctypes.c_int
        iokit.IOServiceGetMatchingServices.argtypes = [u32, vp, ctypes.POINTER(u32)]
        iokit.IOIteratorNext.restype = u32
        iokit.IOIteratorNext.argtypes = [u32]
        iokit.IOObjectRelease.argtypes = [u32]
        iokit.IORegistryEntryCreateCFProperty.restype = vp
        iokit.IORegistryEntryCreateCFProperty.argtypes = [u32, vp, vp, u32]
        iokit.IORegistryEntrySearchCFProperty.restype = vp
        iokit.IORegistryEntrySearchCFProperty.argtypes = [u32, ctypes.c_char_p, vp, vp, u32]
        self._cf, self._iokit = cf, iokit
        self._keys = {k: cf.CFStringCreateWithCString(None, k.encode(), self._UTF8) for k in (
            "Statistics", "Bytes (Read)", "Bytes (Write)", "Protocol Characteristics",
            "Physical Interconnect Location")}
        r, w, n = self._read()
        if n == 0:
            raise OSError("no internal disk found in IOKit")

    def _number(self, d, key):
        num = self._cf.CFDictionaryGetValue(d, self._keys[key])
        out = ctypes.c_int64(0)
        if num:
            self._cf.CFNumberGetValue(num, self._SINT64, ctypes.byref(out))
        return out.value

    def _is_internal(self, driver):
        proto = self._iokit.IORegistryEntrySearchCFProperty(
            driver, b"IOService", self._keys["Protocol Characteristics"], None,
            self._SEARCH_PARENTS)
        if not proto:
            return False
        try:
            loc = self._cf.CFDictionaryGetValue(proto, self._keys["Physical Interconnect Location"])
            buf = ctypes.create_string_buffer(64)
            return bool(loc) and self._cf.CFStringGetCString(loc, buf, 64, self._UTF8) \
                and buf.value == b"Internal"
        finally:
            self._cf.CFRelease(proto)

    def _read(self):
        it = ctypes.c_uint32(0)
        matching = self._iokit.IOServiceMatching(b"IOBlockStorageDriver")  # consumed below
        if self._iokit.IOServiceGetMatchingServices(0, matching, ctypes.byref(it)) != 0:
            raise OSError("IOServiceGetMatchingServices failed")
        read = written = disks = 0
        try:
            while True:
                driver = self._iokit.IOIteratorNext(it.value)
                if not driver:
                    break
                try:
                    if not self._is_internal(driver):
                        continue
                    stats = self._iokit.IORegistryEntryCreateCFProperty(
                        driver, self._keys["Statistics"], None, 0)
                    if not stats:
                        continue
                    try:
                        read += self._number(stats, "Bytes (Read)")
                        written += self._number(stats, "Bytes (Write)")
                        disks += 1
                    finally:
                        self._cf.CFRelease(stats)
                finally:
                    self._iokit.IOObjectRelease(driver)
        finally:
            self._iokit.IOObjectRelease(it.value)
        return read, written, disks

    @property
    def kinds(self):
        return {"internal": "apple_ssd" if _is_apple_silicon() else None}

    def __call__(self):
        """{"internal": (bytes_read, bytes_written)} by the internal disks since boot."""
        read, written, _ = self._read()
        return {"internal": (read, written)}


def _linux_disk_kind_for_path(path=".", sys_block="/sys/block", sys_dev="/sys/dev/block"):
    """Type ("nvme", "ssd" or "hdd") of the physical disk that holds `path`, following
    partitions and LVM/LUKS/RAID layers down to a disk. None if it cannot be told."""
    try:
        kinds = _LinuxSystemDiskCounters.disk_kinds(sys_block)
        dev = os.stat(path).st_dev
        node = os.path.realpath(f"{sys_dev}/{os.major(dev)}:{os.minor(dev)}")
    except (OSError, ValueError):
        return None
    for _ in range(8):  # a few layers at most (partition -> dm -> md -> disk)
        name = os.path.basename(node)
        if name in kinds:
            return kinds[name]
        parent = os.path.basename(os.path.dirname(node))
        if parent in kinds:  # a partition: .../block/sda/sda1
            return kinds[parent]
        try:
            slaves = sorted(os.listdir(os.path.join(node, "slaves")))
        except OSError:
            slaves = []
        if not slaves:
            break
        node = os.path.realpath(os.path.join(node, "slaves", slaves[0]))
    kinds_found = set(kinds.values())
    return kinds_found.pop() if len(kinds_found) == 1 else None


class _LinuxSystemDiskCounters:
    """System-wide bytes read/written per physical disk (all processes), from /proc/diskstats.
    No sudo needed. Only whole physical disks are counted: partitions, loop devices, device
    mapper (LVM, LUKS), software RAID and RAM disks are skipped, since their I/O is already
    counted on the disks underneath. Each disk's type (nvme, ssd or hdd) is read from sysfs."""

    _VIRTUAL = ("loop", "ram", "zram", "dm-", "md", "sr", "nbd", "fd")
    _SECTOR = 512  # /proc/diskstats always counts 512-byte sectors

    def __init__(self, diskstats="/proc/diskstats", sys_block="/sys/block"):
        self.diskstats = diskstats
        self.kinds = self.disk_kinds(sys_block)
        if not self.kinds:
            raise OSError(f"no physical disks found in {sys_block}")
        self()  # fails here, not mid-measurement, if /proc/diskstats is unreadable

    @classmethod
    def disk_kinds(cls, sys_block="/sys/block"):
        """{disk: "nvme" | "ssd" | "hdd"} for each physical disk."""
        kinds = {}
        for name in sorted(os.listdir(sys_block)):
            path = os.path.join(sys_block, name)
            if name.startswith(cls._VIRTUAL) or not os.path.exists(os.path.join(path, "device")):
                continue
            try:
                with open(os.path.join(path, "queue", "rotational")) as f:
                    rotational = f.read().strip() == "1"
            except OSError:
                rotational = False
            kinds[name] = "nvme" if name.startswith("nvme") else (
                "hdd" if rotational else "ssd")
        return kinds

    def __call__(self):
        """{disk: (bytes_read, bytes_written)} since boot."""
        result = {}
        with open(self.diskstats) as f:
            for line in f:
                fields = line.split()
                if len(fields) >= 10 and fields[2] in self.kinds:
                    result[fields[2]] = (int(fields[5]) * self._SECTOR,
                                         int(fields[9]) * self._SECTOR)
        return result


class _IOReportEnergyCounter:
    """Reads cumulative energy counters from the macOS IOReport "Energy Model" group, e.g. the
    "DRAM" channel on Apple Silicon. These are the kernel's own energy counters (the same ones
    tools like macmon read); they need no sudo. IOReport is a private Apple API, accessed here
    through ctypes, so every failure is reported and degrades to "unavailable".

    Usage: counter.snapshot() at begin and at end, then counter.joules_between(a, b).
    """

    _UNIT_TO_J = {"mJ": 1e-3, "uJ": 1e-6, "nJ": 1e-9, "J": 1.0}
    _UTF8 = 0x08000100  # kCFStringEncodingUTF8

    def __init__(self, channel_name, group="Energy Model"):
        self.channel_name = channel_name
        vp, i64 = ctypes.c_void_p, ctypes.c_int64
        cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
        ior = ctypes.CDLL("/usr/lib/libIOReport.dylib")

        cf.CFStringCreateWithCString.restype = vp
        cf.CFStringCreateWithCString.argtypes = [vp, ctypes.c_char_p, ctypes.c_uint32]
        cf.CFStringGetCString.argtypes = [vp, ctypes.c_char_p, i64, ctypes.c_uint32]
        cf.CFDictionaryGetValue.restype = vp
        cf.CFDictionaryGetValue.argtypes = [vp, vp]
        cf.CFArrayGetCount.restype = i64
        cf.CFArrayGetCount.argtypes = [vp]
        cf.CFArrayGetValueAtIndex.restype = vp
        cf.CFArrayGetValueAtIndex.argtypes = [vp, i64]
        cf.CFRelease.argtypes = [vp]
        ior.IOReportCopyChannelsInGroup.restype = vp
        ior.IOReportCopyChannelsInGroup.argtypes = [vp, vp, ctypes.c_uint64, ctypes.c_uint64,
                                                    ctypes.c_uint64]
        ior.IOReportCreateSubscription.restype = vp
        ior.IOReportCreateSubscription.argtypes = [vp, vp, ctypes.POINTER(vp), ctypes.c_uint64,
                                                   vp]
        ior.IOReportCreateSamples.restype = vp
        ior.IOReportCreateSamples.argtypes = [vp, vp, vp]
        ior.IOReportCreateSamplesDelta.restype = vp
        ior.IOReportCreateSamplesDelta.argtypes = [vp, vp, vp]
        ior.IOReportChannelGetChannelName.restype = vp
        ior.IOReportChannelGetChannelName.argtypes = [vp]
        ior.IOReportChannelGetUnitLabel.restype = vp
        ior.IOReportChannelGetUnitLabel.argtypes = [vp]
        ior.IOReportSimpleGetIntegerValue.restype = i64
        ior.IOReportSimpleGetIntegerValue.argtypes = [vp, ctypes.c_int32]
        self._cf, self._ior = cf, ior

        self._key_channels = self._cfstr("IOReportChannels")
        group_ref = self._cfstr(group)
        channels = ior.IOReportCopyChannelsInGroup(group_ref, None, 0, 0, 0)
        cf.CFRelease(group_ref)
        if not channels:
            raise OSError(f"IOReport group '{group}' not found")
        self._sub_dict = vp()
        self._sub = ior.IOReportCreateSubscription(None, channels, ctypes.byref(self._sub_dict),
                                                   0, None)
        cf.CFRelease(channels)
        if not self._sub:
            raise OSError("could not create IOReport subscription")

        # Fail early if the channel does not exist on this chip.
        a = self.snapshot()
        try:
            if self.joules_between(a, a) is None:
                raise OSError(f"IOReport channel '{channel_name}' not found")
        finally:
            self.release(a)

    def _cfstr(self, text):
        return self._cf.CFStringCreateWithCString(None, text.encode(), self._UTF8)

    def _pystr(self, cfstr):
        if not cfstr:
            return ""
        buf = ctypes.create_string_buffer(256)
        self._cf.CFStringGetCString(cfstr, buf, 256, self._UTF8)
        return buf.value.decode()

    def snapshot(self):
        """Take a sample of all subscribed counters (release it with release())."""
        sample = self._ior.IOReportCreateSamples(self._sub, self._sub_dict, None)
        if not sample:
            raise OSError("IOReportCreateSamples failed")
        return sample

    def release(self, sample):
        if sample:
            self._cf.CFRelease(sample)

    def joules_between(self, a, b):
        """Energy (J) of the channel between samples a and b, or None if it is not present."""
        delta = self._ior.IOReportCreateSamplesDelta(a, b, None)
        if not delta:
            raise OSError("IOReportCreateSamplesDelta failed")
        try:
            channels = self._cf.CFDictionaryGetValue(delta, self._key_channels)
            for i in range(self._cf.CFArrayGetCount(channels) if channels else 0):
                ch = self._cf.CFArrayGetValueAtIndex(channels, i)
                if self._pystr(self._ior.IOReportChannelGetChannelName(ch)) != self.channel_name:
                    continue
                unit = self._pystr(self._ior.IOReportChannelGetUnitLabel(ch)).strip()
                if unit not in self._UNIT_TO_J:
                    raise OSError(f"unexpected unit '{unit}' for IOReport channel "
                                  f"'{self.channel_name}'")
                return self._ior.IOReportSimpleGetIntegerValue(ch, 0) * self._UNIT_TO_J[unit]
            return None
        finally:
            self._cf.CFRelease(delta)


class _RaplCounters:
    """CPU package and DRAM energy from the Linux powercap RAPL interface
    (/sys/class/powercap/intel-rapl:*; AMD CPUs show up there too).

    Each `energy_uj` file is a cumulative counter that wraps back to 0 after
    `max_energy_range_uj` (often ~262 kJ: about 45 min at 100 W, less on bigger CPUs). Reading
    only at begin() and end() would silently lose whole wraps, so a background thread reads the
    counters every `interval_s` seconds (well under one wrap period) and adds up the deltas,
    correcting each one that went backwards.
    """

    def __init__(self, interval_s=1.0, root="/sys/class/powercap"):
        self.interval_s = interval_s
        self.domains = {"package": [], "dram": []}  # per socket: (energy_uj path, wrap range)
        packages = []
        for name in os.listdir(root):
            m = re.fullmatch(r"intel-rapl:(\d+)", name)
            if not m:
                continue
            path = os.path.join(root, name)
            if not self._read_name(path).startswith("package"):
                continue  # e.g. "psys", the whole platform
            dram = None
            for sub in sorted(os.listdir(path)):
                sub_path = os.path.join(path, sub)
                if sub.startswith(name + ":") and self._read_name(sub_path) == "dram":
                    dram = self._domain(sub_path)
            packages.append((int(m.group(1)), self._domain(path), dram))
        if not packages:
            raise OSError(f"no RAPL package domains under {root}")
        for _, package, dram in sorted(packages):
            self.domains["package"].append(package)
            if dram is not None:
                self.domains["dram"].append(dram)
        for domain in self.domains["package"] + self.domains["dram"]:
            self._read_uj(domain[0])  # fails here, not mid-measurement, if not readable

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._last = {}
        self._total_uj = {}

    @staticmethod
    def _read_name(path):
        try:
            with open(os.path.join(path, "name")) as f:
                return f.read().strip()
        except OSError:
            return ""

    @staticmethod
    def _read_uj(path):
        with open(path) as f:
            return int(f.read())

    def _domain(self, path):
        with open(os.path.join(path, "max_energy_range_uj")) as f:
            wrap = int(f.read())
        return os.path.join(path, "energy_uj"), wrap

    def _sample(self):
        with self._lock:
            for kind, domains in self.domains.items():
                for i, (path, wrap) in enumerate(domains):
                    value = self._read_uj(path)
                    delta = value - self._last[kind, i]
                    if delta < 0:  # the counter wrapped around since the last read
                        delta += wrap
                    self._total_uj[kind, i] += delta
                    self._last[kind, i] = value

    def _run(self):
        while not self._stop.wait(self.interval_s):
            try:
                self._sample()
            except OSError:
                pass  # a missed read is fine; the next one covers the gap (if under one wrap)

    def begin(self):
        with self._lock:
            for kind, domains in self.domains.items():
                for i, (path, _) in enumerate(domains):
                    self._last[kind, i] = self._read_uj(path)
                    self._total_uj[kind, i] = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="RAPL Sampling Thread",
                                        daemon=True)
        self._thread.start()

    def end(self):
        self._sample()  # first, so the window ends now and not after the thread stops
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.interval_s + 1)
            self._thread = None

    def refresh(self):
        """Read the counters now (only while measuring), so joules() is up to date."""
        if self._thread is not None:
            try:
                self._sample()
            except OSError:
                pass

    def joules(self, kind):
        """Energy per socket (J) for "package" or "dram"; [] if the CPU has no such domain."""
        with self._lock:
            return [self._total_uj.get((kind, i), 0) * 1e-6
                    for i in range(len(self.domains[kind]))]


class ThreadGpuSamplingPyNvml(threading.Thread):
    """
    Thread to sample GPU power draw and utilization using PyNVML.

    This thread can sample one OR MANY GPUs. At each tick it reads the power, utilization and
    temperature of every monitored GPU and stores the samples per GPU index, so that energy can
    later be integrated per GPU and aggregated across the whole system.

    If NVML cannot be initialised or no GPUs are present, the thread degrades gracefully: it
    collects nothing and the meter reports 0 J of GPU energy. No exception is propagated.
    """

    def __init__(self, name, gpu_indices=None, seconds_between_samples=0.1):
        """
        Args:
            name: Thread name
            gpu_indices: Iterable of GPU device indices to monitor. If None (default), ALL
                detected GPUs are monitored.
            seconds_between_samples: Sampling interval in seconds
        """
        threading.Thread.__init__(self)
        self.name = name
        self.daemon = True  # Thread dies when main thread exits
        self.stop = False
        self.seconds_between_samples = seconds_between_samples

        # Thread-safe storage with timestamps.
        self._lock = threading.Lock()
        # samples is keyed by GPU index: {gpu_index: [ {timestamp, power_w, gpu_util, temp_c}, ...]}
        self.samples = {}

        # For backward compatibility (system-wide aggregates per tick).
        self.power_draw_history = []   # total GPU power (sum across GPUs) per tick
        self.activity_history = []     # mean GPU utilization across GPUs per tick

        # Initialize NVML and resolve which GPUs to track.
        self._nvml_initialized = False
        self.gpu_handles = {}   # {gpu_index: handle}
        self.gpu_indices = []

        if not _PYNVML_AVAILABLE:
            # No NVIDIA stack on this host - that's fine, just nothing to do.
            print("Info: pynvml not available; NVIDIA GPU energy will not be tracked.")
            return

        try:
            nvmlInit()
            self._nvml_initialized = True

            device_count = nvmlDeviceGetCount()

            if device_count == 0:
                print("Info: No NVIDIA GPUs detected; GPU energy will not be tracked.")
                return

            if gpu_indices is None:
                # Default behaviour: track every GPU in the system.
                resolved = list(range(device_count))
            else:
                resolved = [i for i in gpu_indices if 0 <= i < device_count]
                skipped = [i for i in gpu_indices if i < 0 or i >= device_count]
                if skipped:
                    print(
                        f"Warning: GPU indices {skipped} are out of range "
                        f"(only {device_count} GPU(s) detected) and will be skipped."
                    )

            for i in resolved:
                self.gpu_handles[i] = nvmlDeviceGetHandleByIndex(i)
                self.samples[i] = []

            self.gpu_indices = resolved

            if self.gpu_indices:
                print(f"GPU monitoring enabled for device index/indices: {self.gpu_indices}")

        except NVMLError as e:
            # No GPU / driver / permissions - degrade gracefully, never raise.
            print(f"Info: NVML unavailable ({e}); NVIDIA GPU energy will not be tracked.")
        except Exception as e:  # ultra-defensive: never let GPU setup break the meter
            print(f"Info: Unexpected error during GPU setup ({e}); GPU energy will not be tracked.")

    def run(self):
        """Sample GPU metrics for every monitored GPU until self.stop is set to True."""
        if not self._nvml_initialized or not self.gpu_indices:
            return  # nothing to sample; exit quietly

        while not self.stop:
            timestamp = time.time()
            total_power = 0.0
            tick_utils = []

            for i in self.gpu_indices:
                try:
                    handle = self.gpu_handles[i]
                    power_w = nvmlDeviceGetPowerUsage(handle) / 1000.0
                    gpu_util = nvmlDeviceGetUtilizationRates(handle).gpu
                    temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)

                    sample = {
                        'timestamp': timestamp,
                        'power_w': power_w,
                        'gpu_util': gpu_util,
                        'temp_c': temp,
                    }
                    with self._lock:
                        self.samples[i].append(sample)

                    total_power += power_w
                    tick_utils.append(gpu_util)
                except NVMLError as e:
                    print(f"NVML sampling error on GPU {i}: {e}")

            with self._lock:
                self.power_draw_history.append(total_power)
                if tick_utils:
                    self.activity_history.append(float(np.mean(tick_utils)))

            time.sleep(self.seconds_between_samples)

    def get_samples(self, gpu_index=None):
        """Get a copy of collected samples (thread-safe).
        If gpu_index is None, returns {gpu_index: [samples...]}; otherwise the list for one GPU.
        """
        with self._lock:
            if gpu_index is None:
                return {i: self.samples[i].copy() for i in self.gpu_indices}
            return self.samples.get(gpu_index, []).copy()

    def clear_samples(self):
        with self._lock:
            for i in self.gpu_indices:
                self.samples[i].clear()
            self.power_draw_history.clear()
            self.activity_history.clear()

    def shutdown(self):
        self.stop = True
        if self.ident is not None:  # started
            self.join(timeout=1.0)
        if self._nvml_initialized:
            try:
                nvmlShutdown()
            except NVMLError:
                pass

    @staticmethod
    def _compute_stats(samples):
        if not samples:
            return None
        power_values = np.array([s['power_w'] for s in samples])
        util_values = np.array([s['gpu_util'] for s in samples])
        temp_values = np.array([s['temp_c'] for s in samples])
        timestamps = np.array([s['timestamp'] for s in samples])

        if len(timestamps) > 1:
            energy_j = float(np.trapezoid(power_values, timestamps))
        else:
            energy_j = 0.0

        return {
            'power_w': {
                'mean': np.mean(power_values),
                'max': np.max(power_values),
                'min': np.min(power_values),
                'std': np.std(power_values),
            },
            'gpu_util': {
                'mean': np.mean(util_values),
                'max': np.max(util_values),
                'min': np.min(util_values),
            },
            'temp_c': {
                'mean': np.mean(temp_values),
                'max': np.max(temp_values),
                'min': np.min(temp_values),
            },
            'duration_s': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
            'num_samples': len(samples),
            'energy_wh': energy_j / 3600,
            'energy_j': energy_j,
        }

    def get_statistics_per_gpu(self):
        with self._lock:
            return {i: self._compute_stats(self.samples[i].copy()) for i in self.gpu_indices}

    def get_statistics(self):
        per_gpu = self.get_statistics_per_gpu()
        valid = {i: s for i, s in per_gpu.items() if s is not None}
        if not valid:
            return None

        total_energy_j = sum(s['energy_j'] for s in valid.values())
        combined_mean_power = sum(s['power_w']['mean'] for s in valid.values())
        total_samples = sum(s['num_samples'] for s in valid.values())
        max_duration = max(s['duration_s'] for s in valid.values())

        return {
            'power_w': {
                'mean': combined_mean_power,
                'max': sum(s['power_w']['max'] for s in valid.values()),
                'min': sum(s['power_w']['min'] for s in valid.values()),
            },
            'gpu_util': {
                'mean': float(np.mean([s['gpu_util']['mean'] for s in valid.values()])),
                'max': max(s['gpu_util']['max'] for s in valid.values()),
            },
            'temp_c': {
                'mean': float(np.mean([s['temp_c']['mean'] for s in valid.values()])),
                'max': max(s['temp_c']['max'] for s in valid.values()),
            },
            'duration_s': max_duration,
            'num_samples': total_samples,
            'num_gpus': len(valid),
            'energy_wh': total_energy_j / 3600,
            'energy_j': total_energy_j,
            'per_gpu': per_gpu,
        }


class ThreadAppleSiliconPowerSampling(threading.Thread):
    """
    Samples on-die power on Apple Silicon (arm64) Macs using the `powermetrics` CLI tool, which
    is the Apple-supported way to read SoC power (there is no RAPL/MSR interface on Apple
    Silicon).

    powermetrics emits one block per sampling interval, e.g.:

        *** Sampled system activity (... ) (1004.97ms elapsed) ***
        ...
        CPU Power: 102 mW
        GPU Power: 7 mW
        ANE Power: 0 mW
        DRAM Power: 178 mW
        Package Power: 339 mW

    We parse EVERY `<Name> Power: <value> <unit>` line so we capture CPU, GPU, ANE, DRAM,
    Package and any cluster-level fields the chip reports. Each block reports the AVERAGE power
    over the interval, and the header reports the actual elapsed time, so energy for a field is
    accumulated as power * elapsed over the blocks where that field appears.

    Notes / caveats:
      - `powermetrics` requires root. Run your script with sudo (or configure passwordless sudo).
      - Field availability is chip/OS-dependent. On recent chips (e.g. M3) powermetrics reports
        no DRAM power at all, which is why EnergyMeter reads DRAM from IOReport instead.
    """

    # `<field> Power: <number> mW|W` ; field allows letters, digits, spaces, + - / ( )
    POWER_RE = re.compile(r"^\s*([A-Za-z0-9 +\-/()]+?)\s+Power:\s*([0-9]+(?:\.[0-9]+)?)\s*(mW|W)\b")
    # header line carries the true elapsed time for the block
    ELAPSED_RE = re.compile(r"\(([0-9]+(?:\.[0-9]+)?)\s*ms elapsed\)")
    # GPU activity, from the gpu_power sampler: share of the interval the GPU was busy (like
    # NVML's utilization) and its average clock while busy.
    GPU_RESIDENCY_RE = re.compile(r"^\s*GPU HW active residency:\s*([0-9]+(?:\.[0-9]+)?)\s*%")
    GPU_FREQUENCY_RE = re.compile(r"^\s*GPU HW active frequency:\s*([0-9]+(?:\.[0-9]+)?)\s*MHz")

    def __init__(self, name, interval_ms=100, samplers="cpu_power,gpu_power",
                 extra_powermetrics_args=None):
        threading.Thread.__init__(self)
        self.name = name
        self.daemon = True
        self.stop = False
        self.interval_ms = max(1, int(interval_ms))
        self.interval_s = self.interval_ms / 1000.0
        self.samplers = samplers
        self.extra_powermetrics_args = extra_powermetrics_args or []

        self._lock = threading.Lock()
        # Notified every time a new powermetrics block header is read.
        self._new_block = threading.Condition(self._lock)
        # samples: [ {timestamp, interval_s, powers: {field: watts},
        #             gpu_active_residency_pct, gpu_active_frequency_mhz}, ... ]
        # `timestamp` is when the block was emitted, i.e. the END of the interval it averages,
        # so a block covers [timestamp - interval_s, timestamp].
        self.samples = []
        # Measurement window set by EnergyMeter.begin()/end(); only the part of each block that
        # overlaps it is counted. None means "open" on that side.
        self.window_start = None
        self.window_end = None

        self.popen = None
        self.pgid = None

    @staticmethod
    def _normalize_field(name):
        key = name.strip().lower()
        for ch in [" ", "-", "/", "+", "(", ")"]:
            key = key.replace(ch, "_")
        while "__" in key:
            key = key.replace("__", "_")
        return key.strip("_")

    def run(self):
        cmd = [
            "sudo", "powermetrics",
            "--samplers", self.samplers,
            "-i", str(self.interval_ms),
        ] + list(self.extra_powermetrics_args)

        try:
            self.popen = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                preexec_fn=os.setpgrp,  # own process group so we can kill the sudo'd child
            )
            self.pgid = os.getpgid(self.popen.pid)
        except Exception as e:
            print(f"Warning: Could not start powermetrics: {e}")
            print("Apple Silicon power measurements will not be available.")
            return

        current = None  # the block currently being accumulated
        try:
            for line in self.popen.stdout:
                if self.stop:
                    break

                em = self.ELAPSED_RE.search(line)
                if em:
                    # New block. powermetrics prints a whole block at once at the END of its
                    # interval, so the header's arrival time is the interval's end time.
                    current = {
                        'timestamp': time.time(),
                        'interval_s': float(em.group(1)) / 1000.0,
                        'powers': {},
                        'gpu_active_residency_pct': None,
                        'gpu_active_frequency_mhz': None,
                    }
                    with self._new_block:
                        self.samples.append(current)
                        self._new_block.notify_all()
                    continue

                if current is not None and line.lstrip().startswith("GPU HW active"):
                    gm = self.GPU_RESIDENCY_RE.search(line)
                    if gm:
                        with self._lock:
                            current['gpu_active_residency_pct'] = float(gm.group(1))
                        continue
                    gm = self.GPU_FREQUENCY_RE.search(line)
                    if gm:
                        with self._lock:
                            current['gpu_active_frequency_mhz'] = float(gm.group(1))
                        continue

                pm = self.POWER_RE.search(line)
                if pm and current is not None:
                    field = self._normalize_field(pm.group(1))
                    value = float(pm.group(2))
                    unit = pm.group(3)
                    watts = value / 1000.0 if unit == "mW" else value
                    with self._lock:
                        current['powers'][field] = watts
        except Exception as e:
            print(f"Warning: powermetrics sampling stopped: {e}")
        finally:
            with self._new_block:
                self._new_block.notify_all()  # wake any waiter if powermetrics died

    def wait_for_block_after(self, t, timeout):
        """Block until powermetrics has emitted a block that ends after time `t` (plus a short
        grace period so that block's power lines are parsed). Returns False on timeout or if
        powermetrics is not running."""
        deadline = time.time() + timeout
        with self._new_block:
            while not (self.samples and self.samples[-1]['timestamp'] > t):
                remaining = deadline - time.time()
                if remaining <= 0 or not self.is_alive():
                    return False
                self._new_block.wait(min(remaining, 0.1))
        time.sleep(0.02)  # the block's lines arrive together with its header
        return True

    def _overlap_s(self, s):
        """Seconds of block `s` that fall inside the measurement window."""
        end = s['timestamp']
        start = end - s['interval_s']
        if self.window_start is not None:
            start = max(start, self.window_start)
        if self.window_end is not None:
            end = min(end, self.window_end)
        return max(0.0, end - start)

    def get_samples_in_window(self, field):
        """Blocks that overlap the measurement window and report `field`, oldest first:
        [{timestamp, interval_s, overlap_s, power_w, utilization_pct, frequency_mhz}, ...].
        `timestamp` is the end of the interval each block averages. utilization_pct and
        frequency_mhz are the GPU's (None for other fields, or if powermetrics did not say)."""
        with self._lock:
            out = []
            for s in self.samples:
                overlap = self._overlap_s(s)
                if field not in s['powers'] or overlap <= 0:
                    continue
                gpu = field == "gpu"
                out.append({
                    'timestamp': s['timestamp'],
                    'interval_s': s['interval_s'],
                    'overlap_s': overlap,
                    'power_w': s['powers'][field],
                    'utilization_pct': s.get('gpu_active_residency_pct') if gpu else None,
                    'frequency_mhz': s.get('gpu_active_frequency_mhz') if gpu else None,
                })
            return out

    def get_total_joules(self, field):
        """Total energy (joules) for a normalized field key (e.g. 'cpu', 'gpu', 'ane', 'dram',
        'package'), accumulated as sum(power * interval) over blocks where the field appears."""
        with self._lock:
            return float(sum(
                s['powers'][field] * self._overlap_s(s)
                for s in self.samples if field in s['powers']
            ))

    def get_energy_joules_per_field(self):
        """Return {field: joules} for every field observed across the run."""
        with self._lock:
            fields = set()
            for s in self.samples:
                fields.update(s['powers'].keys())
            result = {}
            for f in fields:
                result[f] = float(sum(
                    s['powers'][f] * self._overlap_s(s)
                    for s in self.samples if f in s['powers']
                ))
            return result

    def get_statistics(self):
        with self._lock:
            # Only blocks that overlap the measurement window.
            in_window = [s for s in self.samples if s['powers'] and self._overlap_s(s) > 0]
            if not in_window:
                return None
            fields = set()
            for s in in_window:
                fields.update(s['powers'].keys())

            stats = {'num_samples': len(in_window), 'fields': {}}
            stats['duration_s'] = sum(self._overlap_s(s) for s in in_window)

            for f in fields:
                vals = np.array([s['powers'][f] for s in in_window if f in s['powers']])
                energy_j = float(sum(
                    s['powers'][f] * self._overlap_s(s)
                    for s in in_window if f in s['powers']
                ))
                stats['fields'][f] = {
                    'power_w': {
                        'mean': float(np.mean(vals)),
                        'max': float(np.max(vals)),
                        'min': float(np.min(vals)),
                        'std': float(np.std(vals)),
                    },
                    'energy_j': energy_j,
                    'energy_wh': energy_j / 3600,
                    'num_samples': int(vals.size),
                }
            return stats

    def shutdown(self):
        """Stop sampling and terminate the (sudo'd) powermetrics process group."""
        self.stop = True
        if self.popen is not None and self.popen.poll() is None:
            # Signal the `sudo` process itself (it runs with our real uid, so no password is
            # needed); sudo relays the signal to the root-owned powermetrics child. A
            # `sudo kill` here would need a password unless kill is also in sudoers.
            try:
                self.popen.terminate()
                self.popen.wait(timeout=5)
            except Exception:
                pass
        if self.ident is not None:  # started
            self.join(timeout=2.0)


class EnergyMeter:
    """
    The consumption of each component is measured as follows:

    - CPU: on Linux/Intel/AMD via RAPL; on Apple Silicon via `powermetrics`.
    - DRAM: on Linux/Intel via RAPL; on Apple Silicon via the kernel's IOReport "DRAM"
        energy counter (a private Apple API read through ctypes; no sudo needed).
    - GPU: NVIDIA GPUs via PyNVML (all detected GPUs by default, multi-GPU aware); on Apple
        Silicon the integrated GPU via `powermetrics`. ANE (Neural Engine) and package power
        are also captured on Apple Silicon.
    - Disk: tracked by counting bytes read/written, then converting them to energy per disk
        with the speed/power model:
            disk_active_time = (bytes_read + bytes_written) / DISK_SPEED
            disk_idle_time   = total_meter_time - disk_active_time
            total_energy     = disk_active_time * DISK_ACTIVE_POWER (+ idle term if enabled)
        With disk_scope="process" (default), bytes are those of the measured code (bpftrace on
        Linux, which needs passwordless sudo; proc_pid_rusage on macOS). With
        disk_scope="system", bytes are those of every app on the physical disks (/proc/diskstats
        on Linux, IOKit on macOS; no sudo). Any of DISK_SPEED, DISK_ACTIVE_POWER and
        DISK_IDLE_POWER not given defaults to the disk's type (DISK_DEFAULTS).
    """

    #################################### CONSTANTS ####################################
    # Linux: bpftrace counts requested bytes of read()/write() syscalls per command.
    SCRIPT = (
        "tracepoint:syscalls:sys_enter_write {@wbytes[comm] = sum(args->count);} "
        "tracepoint:syscalls:sys_enter_read {@rbytes[comm] = sum(args->count);}"
    )
    ###################################################################################

    def __init__(self, disk_avg_speed=None, disk_active_power=None, disk_idle_power=None,
                 label=None, include_idle=False, ignore_disk=False,
                 gpu_index=None, gpu_indices=None, gpu_sampling_rate=0.1,
                 cpu_sampling_rate=0.1, powermetrics_extra_args=None, rapl_sampling_rate=1.0,
                 prometheus_port=None, prometheus_addr="127.0.0.1", disk_scope="process"):
        """
        :param disk_avg_speed: average read+write speed of the disk (bytes/s). Default: by
            disk type (see DISK_DEFAULTS); the same for the two powers below.
        :param disk_active_power: disk power when active (W).
        :param disk_idle_power: disk power when idle (W).
        :param label: optional string to identify the meter.
        :param include_idle: include idle-time energy for disk and GPU.
        :param ignore_disk: if True, disk is not tracked (on Linux: no bpftrace / no sudo needed).
        :param disk_scope: "process" (default) counts disk I/O of the measured code: this
            process on macOS, python processes on Linux (bpftrace, needs passwordless sudo;
            the disk type is that of the disk holding the working directory). "system" counts
            all I/O by any app: the internal disk on macOS (IOKit), every physical disk on
            Linux (/proc/diskstats); no sudo.
        :param gpu_index: (backwards-compatible) a SINGLE NVIDIA GPU index to monitor.
        :param gpu_indices: iterable of NVIDIA GPU indices. If both this and gpu_index are None
            (the default), ALL detected NVIDIA GPUs are tracked.
        :param gpu_sampling_rate: NVIDIA GPU sampling interval in seconds (default 0.1 = 10 Hz).
        :param cpu_sampling_rate: Apple Silicon powermetrics interval in seconds (default 0.1).
            Ignored on non-Apple-Silicon hosts.
        :param powermetrics_extra_args: optional list of extra args passed to powermetrics.
        :param rapl_sampling_rate: seconds between RAPL counter reads on Linux (default 1.0).
            Only needs to be shorter than the time it takes a counter to wrap (minutes).
        :param prometheus_port: if set, serve live energy metrics for Prometheus at
            http://<prometheus_addr>:<prometheus_port>/metrics until cleanup().
        :param prometheus_addr: address the metrics server listens on (default 127.0.0.1,
            this machine only; use 0.0.0.0 to let Prometheus scrape from another host).
        """
        self.label = label if label else "Meter"
        self.prometheus_port = prometheus_port
        self.prometheus_addr = prometheus_addr
        self.prometheus_exporter = None
        self.start_time = self.end_time = None
        self.running = False
        self.include_idle = include_idle
        self.is_apple_silicon = _is_apple_silicon()
        self.is_darwin = _is_darwin()

        # --- CPU / DRAM / Apple power backend ----------------------------------------------
        self.rapl = None                 # RAPL counters (Linux/Intel/AMD)
        self.thread_apple = None         # powermetrics sampler (Apple Silicon)
        self.dram_counter = None         # IOReport DRAM energy counter (Apple Silicon)
        self._dram_begin = self._dram_end = None

        if self.is_apple_silicon:
            self.thread_apple = ThreadAppleSiliconPowerSampling(
                name="Apple Silicon Power Sampling Thread",
                interval_ms=int(cpu_sampling_rate * 1000),
                samplers="cpu_power,gpu_power",
                extra_powermetrics_args=powermetrics_extra_args,
            )
            print(
                "Apple Silicon detected: CPU/GPU/ANE energy will be measured via "
                "`powermetrics` (requires sudo), DRAM via IOReport energy counters."
            )
            # powermetrics does not report DRAM on recent chips (e.g. M3), so DRAM is read
            # from the kernel's IOReport "DRAM" energy counter instead.
            try:
                self.dram_counter = _IOReportEnergyCounter("DRAM")
            except Exception as e:
                print(f"Warning: DRAM energy will not be available (IOReport: {e}).")
        else:
            try:
                self.rapl = _RaplCounters(interval_s=rapl_sampling_rate)
            except Exception as e:
                print(f"Warning: Could not read RAPL energy counters: {e}")
                print("CPU and DRAM energy measurements will not be available (on kernels "
                      ">= 5.10 reading them needs root; see the README).")

        # --- Disk setup ---------------------------------------------------------------------
        if disk_scope not in ("process", "system"):
            raise ValueError('disk_scope must be "process" or "system"')
        self.ignore_disk = ignore_disk
        self.disk_scope = disk_scope
        self.disk_avg_speed = disk_avg_speed
        self.disk_active_power = disk_active_power
        self.disk_idle_power = disk_idle_power
        self._disk_counter = None  # () -> {disk: (bytes_read, bytes_written)}
        self.disk_kinds = {}       # {disk: "apple_ssd" | "nvme" | "ssd" | "hdd" | None}
        self.disk_bytes = {}       # {disk: (bytes_read, bytes_written)} measured, after end()

        if self.ignore_disk:
            self.disk_backend = "none"
        elif disk_scope == "system":
            self.disk_backend = "iokit" if self.is_darwin else "diskstats"
        else:
            self.disk_backend = "rusage" if self.is_darwin else "bpftrace"

        if self.disk_backend == "rusage":
            self._disk_counter = lambda: {"process": _darwin_disk_io_bytes()}
            self.disk_kinds = {"process": "apple_ssd" if self.is_apple_silicon else None}
        elif self.disk_backend == "bpftrace":
            # bpftrace needs root. `sudo -n` never prompts: without passwordless sudo it fails
            # at once (here, not mid-measurement) instead of waiting for a password.
            try:
                ok = subprocess.run(["sudo", "-n", "bpftrace", "--version"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    timeout=10).returncode == 0
            except Exception:
                ok = False
            if ok:
                # Assume the measured code's I/O goes to the disk holding the working directory.
                self.disk_kinds = {"python": _linux_disk_kind_for_path(os.getcwd())}
            else:
                print("Warning: disk tracking per process needs bpftrace with passwordless sudo, "
                      "so the disk will not be tracked. Use disk_scope=\"system\" to track the "
                      "whole disk without sudo, or ignore_disk=True to hide this warning.")
                self.ignore_disk = True
                self.disk_backend = "none"
        elif self.disk_backend in ("iokit", "diskstats"):
            try:
                counter = (_DarwinSystemDiskCounters() if self.disk_backend == "iokit"
                           else _LinuxSystemDiskCounters())
                self._disk_counter, self.disk_kinds = counter, dict(counter.kinds)
            except Exception as e:
                print(f"Warning: disk energy will not be available ({e}).")
                self.ignore_disk = True
                self.disk_backend = "none"

        # Every disk needs a speed and two powers: the ones given, else its type's defaults.
        if self.disk_backend != "none" and None in (disk_avg_speed, disk_active_power,
                                                    disk_idle_power):
            if any(kind not in DISK_DEFAULTS for kind in self.disk_kinds.values()):
                raise Exception("Could not tell this disk's type, so disk_avg_speed, disk_active_power and disk_idle_power are needed to monitor the disk; to ignore the disk, set ignore_disk=True.")
            for disk, kind in self.disk_kinds.items():
                speed, active, idle = self._disk_figures(disk)
                print(f"Disk {disk}: using {'given values and ' if any(v is not None for v in (disk_avg_speed, disk_active_power, disk_idle_power)) else ''}"
                      f"defaults for {kind} ({speed / 1e9:g} GB/s, {active:g} W active, "
                      f"{idle:g} W idle); pass disk_avg_speed, disk_active_power and "
                      f"disk_idle_power for your own.")

        if self.disk_backend == "bpftrace":
            self.disk_command = shlex.split(
                "sudo -n bpftrace -f json -e '{}'".format(EnergyMeter.SCRIPT)
            )
        else:
            self.disk_command = None

        # --- NVIDIA GPU setup ---------------------------------------------------------------
        if gpu_indices is not None:
            resolved_indices = list(gpu_indices)
        elif gpu_index is not None:
            resolved_indices = [gpu_index]
        else:
            resolved_indices = None  # None -> all GPUs

        self.thread_gpu = ThreadGpuSamplingPyNvml(
            name="GPU Sampling Thread",
            gpu_indices=resolved_indices,
            seconds_between_samples=gpu_sampling_rate,
        )

        # Started here, before anything is measured, so a busy port fails fast and cleanly.
        if prometheus_port is not None:
            self.start_prometheus_exporter(prometheus_port, prometheus_addr)

    # ---------------------------------------------------------------------------------------
    def begin(self):
        """Begin measuring. Starts the relevant backends and records the start time."""
        self.start_time = time.time()

        # CPU / DRAM / Apple.
        if self.is_apple_silicon:
            if self.thread_apple is not None:
                self.thread_apple.start()
                # powermetrics needs ~0.5 s to start and only reports at the end of each
                # interval. Wait for its first block so sampling is live, then open the
                # measurement window now: work done right after begin() is never missed.
                startup_timeout = 5.0 + 2 * self.thread_apple.interval_s
                if not self.thread_apple.wait_for_block_after(0, timeout=startup_timeout):
                    print("Warning: powermetrics produced no samples (is passwordless sudo "
                          "configured for it?). CPU/GPU/ANE/DRAM energy will be 0.")
                self.start_time = time.time()
                self.thread_apple.window_start = self.start_time
            if self.dram_counter is not None:
                self.dram_counter.release(self._dram_begin)
                self.dram_counter.release(self._dram_end)
                self._dram_end = None
                self._dram_begin = self.dram_counter.snapshot()
        else:
            if self.rapl:
                self.rapl.begin()
            else:
                print("RAPL is not accessible, no CPU or memory energy metrics are available!")

        # Disk (bpftrace on Linux, kernel byte counters on macOS).
        if self._disk_counter is not None:
            try:
                self._disk_io_start = self._disk_counter()
            except Exception as e:
                print(f"Warning: Could not read disk I/O counters: {e}")
                self.ignore_disk = True
                self.disk_backend = "none"
                self._disk_counter = None
        elif self.disk_backend == "bpftrace":
            try:
                self.popen = subprocess.Popen(
                    self.disk_command, stdout=subprocess.PIPE, preexec_fn=os.setpgrp
                )
                self.disk_pgid = os.getpgid(self.popen.pid)
            except Exception as e:
                print(f"Warning: Could not start {self.disk_backend} for disk monitoring: {e}")
                self.ignore_disk = True
                self.disk_backend = "none"

        # NVIDIA GPU thread (no-op if there are no GPUs).
        self.thread_gpu.start()

        self.end_time = None
        self.running = True

    def end(self):
        """Finish the measurements and compute the duration."""
        end_time = time.time()

        # Apple Silicon DRAM: cumulative counter, so one reading at each end of the window.
        if self.dram_counter is not None and self._dram_begin is not None:
            self._dram_end = self.dram_counter.snapshot()

        # macOS disk counters: read them first so nothing done while stopping the other
        # backends is counted.
        if self._disk_counter is not None:
            try:
                self._disk_io_end = self._disk_counter()
            except Exception as e:
                print(f"Warning: Could not read disk I/O counters: {e}")
                self._disk_io_end = None

        # CPU / DRAM / Apple.
        if self.is_apple_silicon:
            if self.thread_apple is not None:
                # Close the window at end_time, then wait for the block that covers it (it is
                # only emitted at the end of its interval) so the tail is not lost.
                self.thread_apple.window_end = end_time
                if self.thread_apple.is_alive():
                    self.thread_apple.wait_for_block_after(
                        end_time, timeout=1.0 + 2 * self.thread_apple.interval_s
                    )
                self.thread_apple.shutdown()
        else:
            if self.rapl:
                self.rapl.end()

        # Stop disk tracer (bpftrace flushes its maps on SIGTERM, the default kill).
        if self.disk_backend == "bpftrace":
            # Signal the `sudo` process itself (it runs as us, so no password is needed); sudo
            # relays SIGTERM to bpftrace, which then prints its byte counts and exits.
            try:
                self.popen.terminate()
                try:
                    self.popen.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    subprocess.run(["sudo", "-n", "kill", str(self.popen.pid)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"Warning: Could not stop {self.disk_backend}: {e}")

        # Stop NVIDIA GPU sampling.
        self.thread_gpu.stop = True
        self.thread_gpu.join(timeout=2.0)

        # Collect disk byte counts.
        if self._disk_counter is not None and self._disk_io_end is not None:
            self.disk_bytes = self._disk_delta(self._disk_io_start, self._disk_io_end)
        elif self.disk_backend == "bpftrace":
            try:
                po = self.popen.stdout.read()
                self.disk_bytes = {"python": self.__preprocess_bpftrace_output(po)}
            except Exception as e:
                print(f"Warning: Could not read disk-tracer output: {e}")
                self.disk_bytes = {}
        else:
            self.disk_bytes = {}
        self.total_rbytes = sum(r for r, _ in self.disk_bytes.values())
        self.total_wbytes = sum(w for _, w in self.disk_bytes.values())

        # On Apple Silicon, end() itself waits for powermetrics; that wait is not part of the
        # measured work, so the window ends when end() was called.
        self.end_time = end_time if self.is_apple_silicon else time.time()
        self.duration = self.end_time - self.start_time
        self.running = False

    def __preprocess_bpftrace_output(self, bpftrace_output):
        """Extract bytes read/written from the bpftrace JSON output (Linux)."""
        bpftrace_output = bpftrace_output.decode()
        if len(bpftrace_output.strip()) > 0:
            po = bpftrace_output.split("\n")
            rbytes = json.loads(po[3]).get("data").get("@rbytes")
            wbytes = json.loads(po[4]).get("data").get("@wbytes")
            total_rbytes = rbytes.get("python", 0) + rbytes.get("python3", 0)
            total_wbytes = wbytes.get("python", 0) + wbytes.get("python3", 0)
        else:
            total_rbytes = 0
            total_wbytes = 0
        return total_rbytes, total_wbytes

    # ---------------------------------------------------------------------------------------
    def get_total_joules_disk(self):
        """Disk energy from the speed/power model. Reference:
        Kansal et al., "Virtual machine power metering and provisioning", ACM SoCC 2010."""
        if self.ignore_disk or self.disk_backend == "none":
            return 0
        return self._disk_joules(self.disk_bytes, self.duration)

    def get_disk_details(self):
        """{disk: {"type", "bytes_read", "bytes_written", "speed", "active_power",
        "idle_power", "joules"}} for each tracked disk, after end()."""
        details = {}
        for disk, (r, w) in self.disk_bytes.items():
            speed, active, idle = self._disk_figures(disk)
            details[disk] = {
                "type": self.disk_kinds.get(disk), "bytes_read": r, "bytes_written": w,
                "speed": speed, "active_power": active, "idle_power": idle,
                "joules": self._disk_joules({disk: (r, w)}, self.duration),
            }
        return details

    @staticmethod
    def _disk_delta(before, after):
        """Bytes moved per disk between two counter readings (disks present in both)."""
        return {d: (max(0, after[d][0] - before[d][0]), max(0, after[d][1] - before[d][1]))
                for d in after if d in before}

    def _disk_figures(self, disk):
        """(speed B/s, active W, idle W) for `disk`: the values given, else its type's."""
        defaults = DISK_DEFAULTS.get(self.disk_kinds.get(disk), {})
        return tuple(given if given is not None else defaults[name] for name, given in (
            ("disk_avg_speed", self.disk_avg_speed),
            ("disk_active_power", self.disk_active_power),
            ("disk_idle_power", self.disk_idle_power)))

    def _disk_joules(self, disk_bytes, duration):
        """Speed/power model, per disk: active time = bytes / speed; the rest of the window
        is idle time (counted only with include_idle)."""
        total = 0.0
        for disk, (r, w) in disk_bytes.items():
            speed, active, idle = self._disk_figures(disk)
            active_s = (r + w) / speed
            total += active_s * active
            if self.include_idle:
                total += max(0.0, duration - active_s) * idle
        return total

    def get_total_joules_cpu(self):
        """Total CPU energy (J), per socket. RAPL on Linux/Intel/AMD; powermetrics on Apple Silicon."""
        if self.is_apple_silicon:
            if self.thread_apple is None:
                return np.array([0])
            joules = self.thread_apple.get_total_joules("cpu")
            if joules == 0.0:
                print("powermetrics recorded no CPU energy (was the script run with sudo?).")
            return np.array([joules])

        if self.rapl:
            return np.array(self.rapl.joules("package"))
        else:
            print("RAPL did not record energy for pkg!")
            return np.array([0])

    def get_total_joules_dram(self):
        """Total DRAM energy (J), per socket. RAPL on Linux/Intel; IOReport's "DRAM" energy counter on
        Apple Silicon (memory controllers, AMCC/DCS, are not included)."""
        if self.is_apple_silicon:
            if self.dram_counter is None or self._dram_begin is None or self._dram_end is None:
                return np.array([0])
            joules = self.dram_counter.joules_between(self._dram_begin, self._dram_end)
            return np.array([joules or 0.0])

        if self.rapl and self.rapl.domains["dram"]:
            return np.array(self.rapl.joules("dram"))
        else:
            print("RAPL did not record energy for dram!")
            return np.array([0])

    def get_total_joules_gpu(self):
        """Total GPU energy (J). On Apple Silicon this is the integrated GPU via powermetrics;
        otherwise it is the sum across all monitored NVIDIA GPUs (trapezoidal integration)."""
        if self.is_apple_silicon:
            if self.thread_apple is None:
                return 0
            return self.thread_apple.get_total_joules("gpu")

        if self.include_idle:
            stats = self.thread_gpu.get_statistics()
            if stats is None or stats.get('num_samples', 0) == 0:
                print("Warning: No GPU samples collected!")
                return 0
            return stats['energy_j']
        else:
            samples_per_gpu = self.thread_gpu.get_samples()
            if not samples_per_gpu:
                print("Warning: No GPU samples collected!")
                return 0
            total_energy_j = 0.0
            for _gpu_index, samples in samples_per_gpu.items():
                active_samples = [s for s in samples if s['gpu_util'] > 0]
                if len(active_samples) > 1:
                    timestamps = np.array([s['timestamp'] for s in active_samples])
                    power_values = np.array([s['power_w'] for s in active_samples])
                    total_energy_j += float(np.trapezoid(power_values, timestamps))
            return total_energy_j

    def get_total_joules_ane(self):
        """Apple Silicon only: total Neural Engine (ANE) energy (J). 0 elsewhere."""
        if self.is_apple_silicon and self.thread_apple is not None:
            return self.thread_apple.get_total_joules("ane")
        return 0

    def get_total_joules_package(self):
        """Apple Silicon only: total package energy (J) - the whole-SoC figure powermetrics
        reports. Useful as a cross-check / upper bound; do not sum it with the per-component
        values (it already includes them). 0 elsewhere."""
        if self.is_apple_silicon and self.thread_apple is not None:
            return self.thread_apple.get_total_joules("package")
        return 0

    def get_total_joules_per_component(self):
        """Total energy (J) per component. On Apple Silicon, ANE is included as an extra
        component (and CPU/GPU/DRAM come from powermetrics)."""
        res = {
            "cpu": self.get_total_joules_cpu(),
            "dram": self.get_total_joules_dram(),
            "gpu": self.get_total_joules_gpu(),
            "disk": self.get_total_joules_disk(),
        }
        if self.is_apple_silicon:
            res["ane"] = self.get_total_joules_ane()
        return res

    # ---- GPU / Apple introspection --------------------------------------------------------
    def get_power_samples(self, component="gpu", gpu_index=0):
        """Power samples taken during the measurement, oldest first, as a list of dicts with the
        same keys on every platform:

            timestamp        Unix time of the sample (on Apple Silicon: end of the interval
                             it averages)
            interval_s       time the sample covers (powermetrics interval; NVIDIA: None)
            power_w          average power in watts
            utilization_pct  GPU busy share in % (Apple: "GPU HW active residency"; NVIDIA:
                             NVML utilization); None for other components
            frequency_mhz    GPU clock while busy (Apple only); None otherwise
            temp_c           GPU temperature (NVIDIA only); None otherwise

        component: "gpu" on both platforms; on Apple Silicon also "cpu", "ane", "package" or any
        other field powermetrics reports. On NVIDIA hosts, gpu_index picks the GPU. Returns []
        when nothing was sampled (e.g. no supported GPU)."""
        if self.is_apple_silicon:
            if self.thread_apple is None:
                return []
            return [{
                'timestamp': s['timestamp'], 'interval_s': s['interval_s'],
                'power_w': s['power_w'], 'utilization_pct': s['utilization_pct'],
                'frequency_mhz': s['frequency_mhz'], 'temp_c': None,
            } for s in self.thread_apple.get_samples_in_window(component)]
        if component != "gpu":
            return []
        return [{
            'timestamp': s['timestamp'], 'interval_s': None, 'power_w': s['power_w'],
            'utilization_pct': s['gpu_util'], 'frequency_mhz': None, 'temp_c': s['temp_c'],
        } for s in self.thread_gpu.get_samples(gpu_index)]

    def get_gpu_statistics(self):
        """Aggregate NVIDIA GPU statistics (with a per-GPU breakdown under 'per_gpu')."""
        return self.thread_gpu.get_statistics()

    def get_gpu_statistics_per_gpu(self):
        return self.thread_gpu.get_statistics_per_gpu()

    def get_total_joules_gpu_per_gpu(self):
        """Per-NVIDIA-GPU energy (J). Respects include_idle. {gpu_index: joules}."""
        samples_per_gpu = self.thread_gpu.get_samples()
        result = {}
        for gpu_index, samples in samples_per_gpu.items():
            relevant = samples if self.include_idle else [s for s in samples if s['gpu_util'] > 0]
            if len(relevant) > 1:
                timestamps = np.array([s['timestamp'] for s in relevant])
                power_values = np.array([s['power_w'] for s in relevant])
                result[gpu_index] = float(np.trapezoid(power_values, timestamps))
            else:
                result[gpu_index] = 0.0
        return result

    def get_apple_statistics(self):
        """Apple Silicon only: detailed per-field power statistics from powermetrics
        (cpu/gpu/ane/dram/package/cluster fields). None elsewhere."""
        if self.is_apple_silicon and self.thread_apple is not None:
            return self.thread_apple.get_statistics()
        return None

    def get_apple_energy_per_field(self):
        """Apple Silicon only: {field: joules} for every field powermetrics reported. {} else."""
        if self.is_apple_silicon and self.thread_apple is not None:
            return self.thread_apple.get_energy_joules_per_field()
        return {}

    # ---- Plotting -------------------------------------------------------------------------
    def plot_total_joules_per_component(self, include_total=True):
        data = self.get_total_joules_per_component()
        if include_total:
            data["total"] = sum(float(np.sum(v)) for v in data.values())
        keys = list(data.keys())
        values = [float(np.sum(val)) for val in data.values()]

        fig, ax = plt.subplots()
        bars = ax.bar(keys, values)
        ax.bar_label(bars)
        plt.xlabel("Components")
        plt.ylabel("Joules")
        plt.title(self.label)
        plt.show()

    def plot_total_joules_per_gpu(self):
        per_gpu = self.get_total_joules_gpu_per_gpu()
        if not per_gpu:
            print("No per-GPU energy data to plot.")
            return
        labels = [f"GPU {i}" for i in per_gpu.keys()]
        values = [float(v) for v in per_gpu.values()]
        fig, ax = plt.subplots()
        bars = ax.bar(labels, values)
        ax.bar_label(bars)
        plt.xlabel("GPU")
        plt.ylabel("Joules")
        plt.title(f"{self.label} - per-GPU energy")
        plt.show()

    # ---- Live totals and Prometheus ------------------------------------------------------
    def get_live_joules_per_component(self):
        """Energy (J) per component since begin(), readable WHILE the meter runs (and equal to
        the final totals after end()). Every value is a float that only grows during a run.
        Disk on Linux (bpftrace) only reports after end(). Never prints or raises."""
        res = {"cpu": 0.0, "dram": 0.0, "gpu": 0.0, "disk": 0.0}
        if self.is_apple_silicon:
            res["ane"] = 0.0
        if self.start_time is None:
            return res
        now = self.end_time if self.end_time is not None else time.time()

        if self.is_apple_silicon:
            if self.thread_apple is not None:
                res["cpu"] = self.thread_apple.get_total_joules("cpu")
                res["gpu"] = self.thread_apple.get_total_joules("gpu")
                res["ane"] = self.thread_apple.get_total_joules("ane")
            if self.dram_counter is not None and self._dram_begin is not None:
                try:
                    if self._dram_end is not None:
                        res["dram"] = self.dram_counter.joules_between(
                            self._dram_begin, self._dram_end) or 0.0
                    else:
                        snap = self.dram_counter.snapshot()
                        try:
                            res["dram"] = self.dram_counter.joules_between(
                                self._dram_begin, snap) or 0.0
                        finally:
                            self.dram_counter.release(snap)
                except Exception:
                    pass
        elif self.rapl:
            self.rapl.refresh()
            res["cpu"] = float(sum(self.rapl.joules("package")))
            res["dram"] = float(sum(self.rapl.joules("dram")))

        if not self.is_apple_silicon:
            try:
                res["gpu"] = float(sum(self.get_total_joules_gpu_per_gpu().values()))
            except Exception:
                pass

        # Disk, from the same speed/power model as get_total_joules_disk().
        disk_bytes = None
        if self.end_time is not None:
            disk_bytes = self.disk_bytes
        elif self._disk_counter is not None and hasattr(self, "_disk_io_start"):
            try:
                disk_bytes = self._disk_delta(self._disk_io_start, self._disk_counter())
            except Exception:
                pass
        if disk_bytes and self.disk_backend != "none" and not self.ignore_disk:
            res["disk"] = self._disk_joules(disk_bytes, now - self.start_time)
        return res

    def start_prometheus_exporter(self, port, addr="127.0.0.1"):
        """Serve live metrics at http://addr:port/metrics (done automatically when the meter
        is created with prometheus_port). Runs until cleanup()."""
        from .prometheus import PrometheusExporter
        if self.prometheus_exporter is None:
            self.prometheus_exporter = PrometheusExporter(self, port, addr)
        return self.prometheus_exporter

    def cleanup(self):
        """Release resources (call when completely done with the meter)."""
        if self.prometheus_exporter is not None:
            self.prometheus_exporter.shutdown()
            self.prometheus_exporter = None
        self.thread_gpu.shutdown()
        if self.thread_apple is not None:
            self.thread_apple.shutdown()
