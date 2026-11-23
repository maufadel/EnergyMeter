# EnergyMeter
EnergyMeter is a Python module that combines RAPL, NVIDIA-SMI and eBPF to estimate the energy consumption of CPU, memory, GPU, and storage on Linux with only three lines of code. This was developed during the development of the article Fadel Argerich, M., & Patiño-Martínez, M. (2024). Measuring and Improving the Energy Efficiency of Large Language Models Inference. IEEE Access.

### Update: version 2.0 is out! 
The new version improves the accuracy of the energy consumption of the GPU while also reducing the overhead of the meter, so there is no visible overhead now.

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

em = EnergyMeter(disk_avg_speed=1600*1e6, # The average speed of your storage (see below how you can get it)
                  disk_active_power=6,    # How many Watts are used when the storage is reading or writing (you can usually find it in specs of your storage)
                  disk_idle_power=1.42,   # How many Watts are used when the storage is idle (you can usually find it in specs of your storage)
                  label="Test Meter",     # A label to identify the measurement
                  include_idle=False)     # If energy used during idle should be accounted for in the measurement. Defaults to False.
    
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

**Disk.** On macOS the CLI also tracks the disk: every byte read or written on the internal disk, by any app (downloads, copies, ...), from the same kernel counters `iostat` uses (no sudo). `stop` shows how many MB were read and written. Disk energy is estimated from those bytes and the disk's speed and power. Apple publishes no power figures for its SSDs, so on Apple Silicon EnergyMeter uses rough defaults (3 GB/s, 3 W active, 0.05 W idle, about 1 J per GB); override any of them with `--disk-speed`, `--disk-active-power` and `--disk-idle-power`, or turn the disk off with `--no-disk`. Intel Macs need all three options. On Linux, disk tracking is off unless you pass all three, and it then only counts I/O of `python` processes.

From Python, the default stays per-process (the code you measure). To count the whole internal disk on macOS, use `EnergyMeter(disk_scope="system")`; on Apple Silicon any disk figures you leave out default to `energymeter.APPLE_SSD_DEFAULTS`.

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
