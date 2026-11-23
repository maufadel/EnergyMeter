"""A small Prometheus exporter for a running EnergyMeter (standard library only).

It serves the Prometheus text format at /metrics:

    energymeter_energy_joules_total{label="...",component="cpu"} 123.4   # counter per component
    energymeter_running{label="..."} 1                                    # 1 between begin/end
    energymeter_start_time_seconds{label="..."} 1.7e9                     # Unix time of begin()
    energymeter_duration_seconds{label="..."} 42.0                        # time measured so far

The energy values are counters: they grow while the meter runs, stop growing at end(), and
restart from 0 on the next begin(), which Prometheus treats as a counter reset. Use
`rate(energymeter_energy_joules_total[1m])` for average power in watts.
"""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _escape(value):
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def render_metrics(meter):
    """The Prometheus text exposition for `meter`, as a string."""
    label = f'label="{_escape(meter.label)}"'
    joules = meter.get_live_joules_per_component()
    if meter.start_time is None:
        duration = 0.0
    else:
        duration = (meter.end_time or time.time()) - meter.start_time

    lines = [
        "# HELP energymeter_energy_joules_total Energy used since the meter's begin(), "
        "per component.",
        "# TYPE energymeter_energy_joules_total counter",
    ]
    for component, value in joules.items():
        lines.append(f'energymeter_energy_joules_total{{{label},component="{component}"}} '
                     f'{float(value):.6f}')
    lines += [
        "# HELP energymeter_running 1 while the meter is measuring (between begin and end).",
        "# TYPE energymeter_running gauge",
        f"energymeter_running{{{label}}} {1 if meter.running else 0}",
        "# HELP energymeter_start_time_seconds Unix time when the measurement began.",
        "# TYPE energymeter_start_time_seconds gauge",
        f"energymeter_start_time_seconds{{{label}}} {meter.start_time or 0:.3f}",
        "# HELP energymeter_duration_seconds Time measured so far.",
        "# TYPE energymeter_duration_seconds gauge",
        f"energymeter_duration_seconds{{{label}}} {duration:.3f}",
    ]
    return "\n".join(lines) + "\n"


class PrometheusExporter:
    """Serves /metrics for one EnergyMeter on a background thread."""

    def __init__(self, meter, port, addr="127.0.0.1"):
        self.meter = meter
        lock = threading.Lock()  # one scrape at a time: reads touch the meter's counters

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.split("?")[0] not in ("/metrics", "/"):
                    self.send_error(404)
                    return
                try:
                    with lock:
                        body = render_metrics(meter).encode()
                except Exception as e:  # never kill the server over one bad scrape
                    self.send_error(500, str(e))
                    return
                self.send_response(200)
                self.send_header("Content-Type", CONTENT_TYPE)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # keep scrapes out of the meter's output
                pass

        self.server = ThreadingHTTPServer((addr, port), Handler)
        self.server.daemon_threads = True
        self.addr, self.port = self.server.server_address[:2]
        self.url = f"http://{addr}:{self.port}/metrics"
        self._thread = threading.Thread(target=self.server.serve_forever,
                                        name="Prometheus Exporter", daemon=True)
        self._thread.start()

    def shutdown(self):
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(timeout=2.0)
