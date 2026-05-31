"""In-process worker that runs the LUVIA pipeline in a QThread.

Replaces the subprocess-based CommandManager for GUI runs. Models cached at
module scope in luvia.tongue.tongue / luvia.straw.straw are reused across
runs in the same process; pipeline events are surfaced as Qt signals; the
GUI's stop button drives a cancellation flag the pipeline checks at stage
boundaries instead of SIGTERM on a subprocess.
"""

import contextlib
import io
import threading
import traceback

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from luvia.arguments import LUVIAargs
from luvia.main import CancelledError, run_from_args


class _StreamToSignal(io.TextIOBase):
    """File-like wrapper that emits each newline-delimited line as a signal."""

    def __init__(self, signal):
        super().__init__()
        self._signal = signal
        self._buffer = ""

    def write(self, text):
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self._signal.emit(line)
        return len(text)

    def flush(self):
        if self._buffer:
            self._signal.emit(self._buffer)
            self._buffer = ""


class PipelineWorker(QObject):
    """Runs a single LUVIA invocation; lives on a QThread."""

    # Each pipeline stage event becomes one signal carrying (name, payload).
    event = pyqtSignal(str, dict)
    # Captured stdout/stderr from the pipeline, one line per emit.
    output_line = pyqtSignal(str)
    # Emitted exactly once at the end with (status, payload).
    # status: "finished" | "cancelled" | "error"
    finished = pyqtSignal(str, dict)

    def __init__(self, argv):
        """argv: list of CLI tokens (e.g. ["main", "--input", "..."]); parsed
        by LUVIAargs and dispatched via run_from_args, so the GUI uses the
        exact same parsing + defaults + validation as the CLI."""
        super().__init__()
        self._argv = list(argv)
        self._cancel = threading.Event()

    def cancel(self):
        """Request cancellation; pipeline checks at the next stage boundary."""
        self._cancel.set()

    def _should_cancel(self):
        return self._cancel.is_set()

    def _on_event(self, name, payload=None):
        self.event.emit(name, dict(payload or {}))

    def run(self):
        stdout = _StreamToSignal(self.output_line)
        stderr = _StreamToSignal(self.output_line)
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                largs = LUVIAargs.main(self._argv)
                run_from_args(largs,
                              on_event=self._on_event,
                              should_cancel=self._should_cancel)
                stdout.flush()
                stderr.flush()
            self.finished.emit("finished", {"command": largs.command})
        except CancelledError as exc:
            stdout.flush()
            stderr.flush()
            self.finished.emit("cancelled", {"reason": str(exc)})
        except SystemExit as exc:
            # argparse calls sys.exit on validation failure; surface as error.
            stdout.flush()
            stderr.flush()
            self.finished.emit("error", {"type": "SystemExit", "message": str(exc)})
        except Exception as exc:
            stdout.flush()
            stderr.flush()
            for line in traceback.format_exc().splitlines():
                self.output_line.emit(line)
            self.finished.emit("error",
                               {"type": type(exc).__name__, "message": str(exc)})


class PipelineRunner(QObject):
    """Owns the QThread + PipelineWorker lifecycle for the GUI."""

    event = pyqtSignal(str, dict)
    output_line = pyqtSignal(str)
    finished = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None
        self._worker = None

    def is_running(self):
        return self._thread is not None and self._thread.isRunning()

    def start(self, argv):
        if self.is_running():
            return False
        self._thread = QThread()
        self._worker = PipelineWorker(argv)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.event.connect(self.event)
        self._worker.output_line.connect(self.output_line)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()
        return True

    def cancel(self):
        if self._worker is not None:
            self._worker.cancel()

    def _on_finished(self, status, payload):
        self.finished.emit(status, payload)
        self._worker = None
        self._thread = None
