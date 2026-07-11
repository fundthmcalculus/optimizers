import os


def pytest_addoption(parser):
    """Add a global flag controlling whether plots are displayed during tests.

    By default the test suite runs fully headless (no windows/browser tabs pop
    up), which keeps it safe for CI and agentic runs. Pass ``--show-plots`` to
    display figures interactively instead.
    """
    parser.addoption(
        "--show-plots",
        action="store_true",
        default=False,
        help="Display plots interactively during tests instead of running headless.",
    )


def pytest_configure(config):
    # Runs after option parsing but before test modules are imported, so setting
    # the env var and backend here takes effect before ``optimizers.plot`` (or
    # any direct ``matplotlib.pyplot`` import) is loaded during collection.
    import matplotlib

    if config.getoption("--show-plots"):
        os.environ["OPTIMIZERS_NO_SHOW"] = "0"
    else:
        os.environ["OPTIMIZERS_NO_SHOW"] = "1"
        matplotlib.use("Agg")
