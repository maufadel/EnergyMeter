"""Command-line interface: measure the energy used by the whole machine between a `start` and a
`stop`, while you do anything else (run programs, use the desktop, ...).

    energymeter start [--label NAME] [--include-idle] [--no-disk] [--disk-speed B/s
                      --disk-active-power W --disk-idle-power W]
                      [--prometheus-port PORT [--prometheus-addr ADDR]]
    energymeter status
    energymeter stop [--json] [--output FILE]

`start` launches a background process that runs an EnergyMeter until `stop` tells it to finish.
CPU, DRAM and GPU (and ANE on Apple Silicon) are read from system-wide counters, so they include
everything running on the machine. The disk is tracked too: all reads and writes by any app,
on the internal disk (macOS) or every physical disk (Linux), with no sudo. Disk energy uses
default figures for each disk's type (Apple SSD, NVMe, SATA SSD, hard drive) unless the
--disk-* options are given; Intel Macs need all three. --no-disk turns it off.

With --prometheus-port, the background process also serves live metrics for Prometheus at
http://ADDR:PORT/metrics until `stop`.
"""
import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

STATE_DIR = Path(os.environ.get("ENERGYMETER_STATE_DIR", Path.home() / ".energymeter"))
SESSION_FILE = STATE_DIR / "session.json"   # written by the daemon while it runs
RESULT_FILE = STATE_DIR / "result.json"     # written by the daemon when it stops
LOG_FILE = STATE_DIR / "daemon.log"         # the meter's own messages go here

START_TIMEOUT_S = 20
STOP_TIMEOUT_S = 30


# ------------------------------------------------------------------------------------- helpers
def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)  # atomic, so readers never see a half-written file


def _running_session():
    """The current session dict if a meter is running, else None (stale files are removed)."""
    session = _read_json(SESSION_FILE)
    if session and _pid_alive(session.get("pid", -1)):
        return session
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()
    return None


def _to_float(value):
    try:
        return float(sum(value))  # RAPL gives one value per CPU socket
    except TypeError:
        return float(value)


def _log_tail(n=15):
    try:
        return "\n".join(LOG_FILE.read_text().splitlines()[-n:])
    except FileNotFoundError:
        return ""


# ------------------------------------------------------------------------------------- daemon
def _daemon(options):
    """Runs in the background process: begin(), wait for SIGTERM/SIGINT, end(), save results."""
    from energymeter import EnergyMeter

    stop_requested = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop_requested.set())

    disk = options.get("disk")
    meter = EnergyMeter(
        label=options["label"],
        include_idle=options["include_idle"],
        ignore_disk=disk is None,
        disk_avg_speed=disk and disk["speed"],
        disk_active_power=disk and disk["active_power"],
        disk_idle_power=disk and disk["idle_power"],
        disk_scope=(disk or {}).get("scope", "process"),
        prometheus_port=options.get("prometheus_port"),
        prometheus_addr=options.get("prometheus_addr") or "127.0.0.1",
    )
    meter.begin()
    _write_json(SESSION_FILE, {
        "pid": os.getpid(),
        "label": options["label"],
        "start_time": meter.start_time,
        "metrics_url": meter.prometheus_exporter and meter.prometheus_exporter.url,
    })
    print(f"[energymeter] started at {time.ctime(meter.start_time)}", flush=True)

    while not stop_requested.wait(timeout=1.0):
        pass

    meter.end()
    components = {k: _to_float(v) for k, v in meter.get_total_joules_per_component().items()}
    result = {
        "label": options["label"],
        "start_time": meter.start_time,
        "end_time": meter.end_time,
        "duration_s": meter.duration,
        "joules": components,
        "total_joules": sum(components.values()),
        "disk_bytes": {"read": int(getattr(meter, "total_rbytes", 0)),
                       "written": int(getattr(meter, "total_wbytes", 0))},
        "disks": meter.get_disk_details(),
    }
    meter.cleanup()
    _write_json(RESULT_FILE, result)
    SESSION_FILE.unlink(missing_ok=True)
    print("[energymeter] stopped", flush=True)


# ------------------------------------------------------------------------------------ commands
def cmd_start(args):
    session = _running_session()
    if session:
        print(f"A meter is already running (\"{session['label']}\", started "
              f"{time.ctime(session['start_time'])}). Run `energymeter stop` first.")
        return 1

    disk_values = (args.disk_speed, args.disk_active_power, args.disk_idle_power)
    intel_mac = sys.platform == "darwin" and platform.machine() != "arm64"
    disk = None
    if args.no_disk:
        pass
    elif not intel_mac or all(v is not None for v in disk_values):
        # The whole disk, by any app. Values not given default to each disk's type.
        disk = {"speed": args.disk_speed, "active_power": args.disk_active_power,
                "idle_power": args.disk_idle_power, "scope": "system"}
    elif any(v is not None for v in disk_values):
        print("On Intel Macs, give all of --disk-speed, --disk-active-power and "
              "--disk-idle-power to track the disk.")
        return 2

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.unlink(missing_ok=True)
    options = {"label": args.label, "include_idle": args.include_idle, "disk": disk,
               "prometheus_port": args.prometheus_port, "prometheus_addr": args.prometheus_addr}

    with open(LOG_FILE, "w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "energymeter.cli", "_daemon", json.dumps(options)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,  # keeps running after this terminal closes
        )

    # Wait until the meter has actually begun (on Apple Silicon this takes ~1 s).
    deadline = time.time() + START_TIMEOUT_S
    while time.time() < deadline:
        session = _read_json(SESSION_FILE)
        if session and session.get("pid") == proc.pid:
            print(f"EnergyMeter started (\"{args.label}\"). Run `energymeter stop` to finish "
                  f"and see the results.")
            if session.get("metrics_url"):
                print(f"Prometheus metrics: {session['metrics_url']}")
            # The meter's own notes (warnings, default disk figures) go to the log; show them.
            for line in _log_tail(50).splitlines():
                if line.startswith(("Warning", "Disk ", "RAPL", "CPU and DRAM")):
                    print("  " + line)
            return 0
        if proc.poll() is not None:
            break
        time.sleep(0.1)

    if proc.poll() is None:
        proc.terminate()
    print("EnergyMeter failed to start. Its output:\n" + _log_tail())
    return 1


def cmd_status(args):
    session = _running_session()
    if not session:
        print("No meter is running.")
        return 1
    elapsed = time.time() - session["start_time"]
    print(f"Meter \"{session['label']}\" running for {elapsed:.0f} s (pid {session['pid']}).")
    if session.get("metrics_url"):
        print(f"Prometheus metrics: {session['metrics_url']}")
    return 0


def cmd_stop(args):
    session = _running_session()
    if not session:
        print("No meter is running. Start one with `energymeter start`.")
        return 1

    os.kill(session["pid"], signal.SIGTERM)
    deadline = time.time() + STOP_TIMEOUT_S
    result = None
    while time.time() < deadline:
        result = _read_json(RESULT_FILE)
        if result or not _pid_alive(session["pid"]):
            result = result or _read_json(RESULT_FILE)
            break
        time.sleep(0.1)
    if not result:
        print("The meter did not report results. Its output:\n" + _log_tail())
        return 1

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    duration = result["duration_s"]
    total = result["total_joules"]
    print(f"EnergyMeter \"{result['label']}\": {duration:.1f} s")
    print(f"  {'component':10s} {'joules':>10s} {'avg watts':>10s}")
    for name, joules in result["joules"].items():
        print(f"  {name:10s} {joules:10.2f} {joules / duration if duration else 0:10.2f}")
    print(f"  {'total':10s} {total:10.2f} {total / duration if duration else 0:10.2f}"
          f"   ({total / 3600:.4f} Wh)")
    if "disks" in result and not result["disks"] and result["joules"].get("disk", 0) == 0:
        print("  disk: not tracked (--no-disk was given, or see the warnings `energymeter start` printed)")
    for name, d in (result.get("disks") or {}).items():
        print(f"  disk {name} ({d['type'] or 'given values'}): {d['bytes_read'] / 1e6:.1f} MB "
              f"read, {d['bytes_written'] / 1e6:.1f} MB written, {d['joules']:.2f} J")
    if args.output:
        print(f"Saved to {args.output}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="energymeter",
        description="Measure the machine's energy use between `start` and `stop`.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="{start,status,stop}")

    p_start = sub.add_parser("start", help="start measuring in the background")
    p_start.add_argument("--label", default="Terminal session", help="name for this measurement")
    p_start.add_argument("--include-idle", action="store_true",
                         help="include idle energy for the disk and NVIDIA GPUs")
    p_start.add_argument("--no-disk", action="store_true", help="do not track the disk")
    p_start.add_argument("--disk-speed", type=float, metavar="BYTES_PER_S",
                         help="average disk read/write speed (default: by disk type)")
    p_start.add_argument("--disk-active-power", type=float, metavar="WATTS",
                         help="disk power while reading/writing (default: by disk type)")
    p_start.add_argument("--disk-idle-power", type=float, metavar="WATTS",
                         help="disk power while idle, used with --include-idle (default: by "
                              "disk type)")
    p_start.add_argument("--prometheus-port", type=int, metavar="PORT",
                         help="serve live metrics for Prometheus on this port until `stop`")
    p_start.add_argument("--prometheus-addr", default="127.0.0.1", metavar="ADDR",
                         help="address for the metrics server (default 127.0.0.1; 0.0.0.0 to "
                              "allow scraping from other hosts)")
    p_start.set_defaults(func=cmd_start)

    p_status = sub.add_parser("status", help="show whether a meter is running")
    p_status.set_defaults(func=cmd_status)

    p_stop = sub.add_parser("stop", help="stop measuring and print the results")
    p_stop.add_argument("--json", action="store_true", help="print the results as JSON")
    p_stop.add_argument("--output", metavar="FILE", help="also save the results as JSON")
    p_stop.set_defaults(func=cmd_stop)

    p_daemon = sub.add_parser("_daemon")  # internal: the background process
    p_daemon.add_argument("options")

    args = parser.parse_args(argv)
    if args.command == "_daemon":
        _daemon(json.loads(args.options))
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
