"""Interruptible background work for the workspace and its dialogs."""

from PySide6.QtCore import QThread, Signal


class WorkspaceJob(QThread):
    completed = Signal(object)
    failed = Signal(str)
    progress = Signal(str)
    counts = Signal(int, int)
    item_ready = Signal(object)

    def __init__(self, work, parent, *, stream=False):
        super().__init__(parent)
        self.work = work
        self.stream = stream

    def run(self):
        try:
            args = (self._report_progress, self.item_ready.emit) if self.stream else (self._report_progress,)
            result = self.work(*args)
            if not self.isInterruptionRequested():
                self.completed.emit(result)
        except InterruptedError:
            return
        except Exception as error:
            self.failed.emit(str(error))

    def _report_progress(self, message, done=None, total=None):
        if self.isInterruptionRequested():
            raise InterruptedError
        self.progress.emit(message)
        if total is not None:
            self.counts.emit(done, total)
