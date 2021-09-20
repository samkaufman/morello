from morello import system_config
from morello.system_config import cpu


def pytest_sessionstart(session):
    # TODO: Ideally, this is set on a per-test/suite basis. No default should be needed.
    system_config.set_current_target(cpu.CpuTarget())
