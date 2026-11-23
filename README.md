# EnergyMeter
EnergyMeter is a Python module that combines RAPL, NVIDIA-SMI and eBPF to estimate the energy consumption of CPU, memory, GPU, and storage on Linux with only three lines of code. This was developed during the development of the article Fadel Argerich, M., & Patiño-Martínez, M. (2024). Measuring and Improving the Energy Efficiency of Large Language Models Inference. IEEE Access.

### Update: version 3.0 is out!
- **macOS on Apple Silicon:** CPU, GPU and Neural Engine energy via `powermetrics`, DRAM via the kernel's energy counters, and disk via the kernel's I/O counters.
- **Terminal command:** `energymeter start` / `status` / `stop` measures the whole machine while you do anything else.
- **Prometheus:** live metrics at `/metrics`, from Python or the terminal.
- **Disk:** default figures for each disk type (Apple SSD, NVMe, SATA SSD, hard drive), so disk values are now optional, and whole-disk tracking without sudo (`disk_scope="system"`).
- **Linux:** RAPL is read directly and sampled every second, so counter wraparounds no longer corrupt long measurements (pyRAPL is no longer a dependency).

Version 2.0 improved the accuracy of GPU energy while reducing the meter's overhead.

## How to install
You can install EnergyMeter via pip or uv and PyPi: `pip install ai-energymeter`, or by cloning this repository and using pip:
```
git clone https://github.com/maufadel/EnergyMeter.git
cd EnergyMeter
pip install .
```

## How to use
The most basic usage of EnergyMeter is as follows:
```
from energymeter import EnergyMeter

em = EnergyMeter(label="Test Meter",     # A label to identify the measurement
                  include_idle=False)     # If energy used during idle should be accounted for in the measurement. Defaults to False.
# Disk energy is estimated from the bytes read and written. By default EnergyMeter uses typical
# figures for your disk's type (see "Disk" below). For better estimates, give your own:
#   EnergyMeter(disk_avg_speed=1600*1e6,  # average speed of your storage in bytes/s (see below how to get it)
#               disk_active_power=6,      # Watts while reading or writing (usually in your storage's specs)
#               disk_idle_power=1.42)     # Watts while idle (usually in your storage's specs)
    
em.begin()
# --> CODE YOU WANT TO MEASURE <--
em.end()

# Plot energy consumption per component.
meter.plot_total_joules_per_component()
# or print(em.get_total_joules_per_component())
```
![Example Output](https://github.com/maufadel/meml/blob/main/example_output.png)

You can check the notebook [Measuring_energy_consumption.ipynb](https://github.com/maufadel/meml/blob/main/Measuring_energy_consumption.ipynb) for more details.

### From the terminal
Installing the package also installs an `energymeter` command. It measures the energy used by the whole machine between a `start` and a `stop`, while you do anything else in the meantime:
```
energymeter start --label "Video call"
# ... use your computer ...
energymeter status          # is a meter running, and for how long?
energymeter stop            # prints energy per component; add --json or --output result.json
```
CPU, DRAM and GPU (and the Neural Engine on Apple Silicon) are read from system-wide counters, so they include everything running on the machine.

**Disk.** The CLI also tracks the disk: every byte read or written by any app (downloads, copies, ...), with no sudo. On macOS it counts the internal disk (the counters `iostat` uses); on Linux, every physical disk from `/proc/diskstats` (partitions, LVM/LUKS, RAID and loop devices are skipped, since their I/O lands on the disks underneath). `stop` shows how many MB each disk read and wrote, and its energy.

Disk energy is estimated from those bytes and each disk's speed and power. Unless you give them, EnergyMeter uses rough figures for the disk's type, which it detects on its own:

| Disk type | Speed | Active power | Idle power |
|---|---|---|---|
| Apple SSD (Apple Silicon) | 3 GB/s | 3 W | 0.05 W |
| NVMe SSD | 2 GB/s | 5 W | 0.5 W |
| SATA SSD | 0.5 GB/s | 3 W | 0.3 W |
| Hard drive | 0.15 GB/s | 6.5 W | 4.5 W |

Override any of them with `--disk-speed`, `--disk-active-power` and `--disk-idle-power` (the values then apply to every disk), or turn the disk off with `--no-disk`. Intel Macs need all three options. Idle power only counts with `--include-idle`.

From Python, the disk values also default to the table above (`energymeter.DISK_DEFAULTS`); any you pass replace them. The library counts only the code you measure by default: the kernel's per-process counters on macOS, and bpftrace on Linux, which needs bpftrace with passwordless sudo (without it the disk is skipped with a warning) and uses the type of the disk holding the working directory. To count the whole disk by any app instead, with no sudo, use `EnergyMeter(disk_scope="system")`. `em.get_disk_details()` shows the bytes, figures and energy per disk.

### Prometheus
EnergyMeter can serve live metrics for Prometheus while it measures, both from Python and from the terminal:
```
em = EnergyMeter(ignore_disk=True, label="LLM inference", prometheus_port=9464)
em.begin()   # metrics are served at http://127.0.0.1:9464/metrics until em.cleanup()
# ...
em.end()
em.cleanup()
```
```
energymeter start --label "Video call" --prometheus-port 9464
energymeter stop   # the metrics server stops with the meter
```
By default the server only listens on this machine (`127.0.0.1`). If Prometheus runs on another host or in Docker, pass `prometheus_addr="0.0.0.0"` (or `--prometheus-addr 0.0.0.0`).

Point Prometheus at it:
```
scrape_configs:
  - job_name: energymeter
    scrape_interval: 5s
    static_configs:
      - targets: ["localhost:9464"]
```
Metrics (each has a `label` label with the meter's label):

| Metric | Type | Meaning |
|---|---|---|
| `energymeter_energy_joules_total{component="cpu\|dram\|gpu\|disk\|ane"}` | counter | Energy used since `begin()`. It stops growing at `end()` and restarts from 0 on the next `begin()`. |
| `energymeter_running` | gauge | 1 between `begin()` and `end()`. |
| `energymeter_start_time_seconds` | gauge | Unix time of `begin()`. |
| `energymeter_duration_seconds` | gauge | Time measured so far. |

Average power in watts is `rate(energymeter_energy_joules_total[1m])`. Disk energy on Linux (bpftrace) only appears after `end()`. With the CLI, the server stops with `energymeter stop`, so the last few seconds before it may not be scraped; read them from `energymeter stop --json` if you need them.

## How to get storage details
You can benchmark your storage speed with the Flexible I/O tester (FIO) as recommended by Google in the following tutorial: https://cloud.google.com/compute/docs/disks/benchmarking-pd-performance 

Storage specs usually include data regarding the power consumption during active (reading or writing) and for idle periods.

## Troubleshooting
### RAPL
CPU and DRAM energy are read from the RAPL counters in `/sys/class/powercap/intel-rapl:*` (AMD CPUs use the same files). Since Linux 5.10 only root can read them. Either run your script (or `energymeter start` and `energymeter stop`) with sudo, or make the counters readable until the next reboot:

`sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj /sys/class/powercap/intel-rapl:*/intel-rapl:*/energy_uj`

The counters wrap around after a fixed amount of energy (often a few minutes to an hour of work), so EnergyMeter reads them every second (`rapl_sampling_rate`) and corrects for wraparounds; long measurements are safe.

### bpftrace
Also note that you need to have bpftrace installed. On ubuntu, you can install it with the following command: 

`sudo apt-get install -y bpftrace`

For other operating systems, please check https://github.com/bpftrace/bpftrace/blob/master/INSTALL.md.

bpftrace runs with `sudo -n`, which never asks for a password, so it needs a passwordless sudo rule. Add one with `sudo visudo -f /etc/sudoers.d/energymeter`, replacing `youruser` and the path (`which bpftrace`) with yours:

`youruser ALL=(root) NOPASSWD: /usr/bin/bpftrace`

Without it, per-process disk tracking is skipped with a warning; `disk_scope="system"` (and the CLI) need no sudo at all.

## Limitations
EnergyMeter requires to be run on bare metal instances running Linux on Intel and NVIDIA hardware. These requirements are inherited from the tools used for tracking the energy consumption: RAPL (Intel), NVIDIA-SMI (NVIDIA), and eBPF (Linux). These tools are known to have a high accuracy thanks to their access to low level sensors, but it is important to keep in mind that the energy consumption metrics provided are still estimations and vary according to different factors including hardware configuration and software versions. Additionally, the energy consumed by cooling, screens and other components not mentioned here are not included in our measurements, so the total energy consumed will likely be different to the total sum of the consumption of CPU, memory, GPU, and storage. 

## Authorship and License
I am developing EnergyMeter as a part of my PhD program at Universidad Politécnica de Madrid. EnergyMeter is open sourced under an MIT License. 

If you found this repository useful, please cite our work:
```
@article{argerich2024measuring,
  title={Measuring and Improving the Energy Efficiency of Large Language Models Inference},
  author={Fadel Argerich, Mauricio and Pati{\~n}o-Mart{\'\i}nez, Marta},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}
```
