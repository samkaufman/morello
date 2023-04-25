import hypothesis

hypothesis.settings.register_profile("dev", deadline=400, print_blob=True)
hypothesis.settings.register_profile(
    "ci", deadline=400, max_examples=300_000, print_blob=True
)
hypothesis.settings.register_profile(
    "fast", deadline=400, max_examples=3000, print_blob=True
)
hypothesis.settings.load_profile("dev")  # Make 'dev' the default


def pytest_sessionstart(session):
    # Import morello inside pytest_sessionstart to make sure it's delayed until after
    # a potential typeguard injection on the command line (e.g., `--typeguard-packages`)
    from morello import system_config
    from morello.system_config import cpu

    system_config.set_current_target(cpu.CpuTarget())
