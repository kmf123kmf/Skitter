import multiprocessing

# In a frozen build (PyInstaller) worker processes start this executable again:
# freeze_support runs their work and exits, before the app (and Qt) loads.
multiprocessing.freeze_support()

from skitter.app import main  # noqa: E402

raise SystemExit(main())
