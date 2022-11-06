import hypothesis

hypothesis.settings.register_profile("ci", max_examples=3000, print_blob=True)
hypothesis.settings.register_profile("debug", deadline=None, print_blob=True)


def pytest_sessionstart(session):
    # Import morello inside pytest_sessionstart to make sure it's delayed until after
    # a potential typeguard injection on the command line (e.g., `--typeguard-packages`)
    from morello import system_config
    from morello.system_config import cpu

    system_config.set_current_target(cpu.CpuTarget())
