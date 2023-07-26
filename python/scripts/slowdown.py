#!/usr/bin/env python3

# TODO: Stop the irqbalance service.
# TODO: Make sure we're booted with isolcpus= set
#   and rcu_nocbs= set.
# TODO: nohz_full=

import pathlib
import functools
import subprocess
import re

from typing import Iterable

_SCALING_GOVERNOR_ROOT = pathlib.Path("/sys/devices/system/cpu")
_GOVERNOR_FILE_RE = re.compile("cpu[0-9]+")


def _using_acpi_cpufreq() -> bool:
    found_any = False
    for inner in _SCALING_GOVERNOR_ROOT.iterdir():
        if inner.is_dir() and _GOVERNOR_FILE_RE.match(inner.name):
            found_any = True
            driver_path = inner / "cpufreq" / "scaling_driver"
            if driver_path.read_text(encoding="utf8").rstrip("\n") != "acpi-cpufreq":
                return False
    if not found_any:
        raise Exception("No scaling_driver files found")
    return True


def _set_boost(enabled: True) -> None:
    boost_path = _SCALING_GOVERNOR_ROOT / "cpufreq" / "boost"
    boost_path.write_text("1" if enabled else "0", encoding="utf8")


def is_cpufreqset_paths() -> tuple[pathlib.Path, pathlib.Path]:
    # TODO: This is sensitive to shell hacking. Secure.
    return tuple(pathlib.Path(subprocess.check_output(["which", n]).decode("utf8").strip())
        for n in ["cpufreq-info", "cpufreq-set"])


def _scaling_governor_paths() -> Iterable[pathlib.Path]:
    for inner in _SCALING_GOVERNOR_ROOT.iterdir():
        if inner.is_dir() and _GOVERNOR_FILE_RE.match(inner.name):
            yield inner / "cpufreq" / "scaling_governor"
    

def set_power_mode(power_mode: str) -> None:
    for gov_path in _scaling_governor_paths():
        gov_path.write_text(power_mode, encoding="utf8")


if __name__ == "__main__":
    assert _using_acpi_cpufreq()
    # set_power_mode("powersave")
    # _set_boost(False)
    
    set_power_mode("ondemand")
    _set_boost(True)
