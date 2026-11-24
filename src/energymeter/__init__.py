from importlib.metadata import PackageNotFoundError, version as _version

from .energy_meter import EnergyMeter, APPLE_SSD_DEFAULTS, DISK_DEFAULTS

try:
    __version__ = _version("ai-energymeter")
except PackageNotFoundError:  # running from a source checkout that is not installed
    __version__ = "unknown"
