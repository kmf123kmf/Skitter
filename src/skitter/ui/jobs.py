"""Background jobs: run long work off the UI thread, reporting through signals.

A job runs `work(progress, cancelled)` in a worker thread. `progress(message,
fraction)` reports status (fraction None or negative: unknown), and
`cancelled()` tells the work to stop early. Signals are emitted from the
worker thread and delivered to receivers on the UI thread (queued).
"""

import contextlib
import logging
import threading
import warnings
from collections.abc import Callable

from PySide6.QtCore import QCoreApplication, QObject, Signal

logger = logging.getLogger(__name__)


class JobCancelled(Exception):
    """Raise from work to stop it as cancelled."""


class Job(QObject):
    progress = Signal(str, float)  # message, fraction (-1: unknown)
    finished = Signal(object)  # the work's result
    failed = Signal(str)
    cancelled = Signal()
    stopped = Signal()  # after finished, failed or cancelled

    def __init__(self, work: Callable, cancel_errors: tuple[type, ...] = (), parent=None):
        super().__init__(parent)
        self._work = work
        self._cancel_errors = (JobCancelled, *cancel_errors)
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> "Job":
        self._thread = threading.Thread(target=self._run, daemon=True, name="skitter-job")
        self._thread.start()
        return self

    def cancel(self) -> None:
        self._cancel.set()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def wait(self, timeout: float | None = None) -> None:
        """Block until the work ends, then deliver its pending signals (tests, shutdown)."""
        if self._thread is not None:
            self._thread.join(timeout)
        QCoreApplication.processEvents()

    def join(self, timeout: float | None = None) -> None:
        """Block until the work's thread has ended, delivering nothing (unlike wait). After
        `stopped` it is about to: then the job may go (with its parent, say) safely."""
        if self._thread is not None:
            self._thread.join(timeout)

    def forget(self) -> None:
        """Done with a stopped job (call it from `stopped`): wait for its thread to end and
        drop its connections and work. Slots that capture their owner (lambdas) would
        otherwise keep it alive through the job (a child it owns): a cycle through C++
        that Python can't collect."""
        self.join()
        for signal in (self.progress, self.finished, self.failed, self.cancelled, self.stopped):
            with warnings.catch_warnings(), contextlib.suppress(RuntimeError, TypeError):
                warnings.simplefilter("ignore")  # nothing connected
                signal.disconnect()
        self._work = None

    def _report(self, message: str, fraction: float | None = None) -> None:
        self.progress.emit(message, -1.0 if fraction is None else float(fraction))

    def _run(self) -> None:
        try:
            result = self._work(self._report, self.is_cancelled)
        except self._cancel_errors:
            self.cancelled.emit()
        except Exception as exc:
            logger.exception("background job failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            if self.is_cancelled():
                self.cancelled.emit()
            else:
                self.finished.emit(result)
        self.stopped.emit()
