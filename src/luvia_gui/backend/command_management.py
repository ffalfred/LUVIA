
from PyQt6.QtCore import QObject, QThread, pyqtSignal
import subprocess
import logging
from datetime import datetime


class CommandWorker(QObject):
    output_line = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, command):
        super().__init__()
        self.command = command
        self.process = None
        self._should_stop = False

    def run(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"{timestamp} - Executing: {self.command}")

        self.process = subprocess.Popen(
            self.command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in self.process.stdout:
            if self._should_stop:
                break
            line = line.strip()
            if line:
                logging.info(line)
                self.output_line.emit(line)

        self.process.wait()
        self.finished.emit()

    def stop(self):
        self._should_stop = True
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.output_line.emit("Process terminated.")


class CommandManager:
    def __init__(self, terminal=None, log_file="command_log.txt", button=None, spinner=None, stop_button=None):
        self.terminal = terminal
        self.log_file = log_file
        self.button = button
        self.spinner = spinner
        self.stop_button = stop_button

        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )

    def execute_command(self, command):
        if self.button:
            self.button.setEnabled(False)
        if self.spinner:
            self.spinner.setVisible(True)

        if hasattr(self, 'stop_button'):
            self.stop_button.setEnabled(True)


        self.thread = QThread()
        self.worker = CommandWorker(command)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.output_line.connect(self.append_output)
        self.worker.finished.connect(self.cleanup)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def append_output(self, line):
        if self.terminal:
            self.terminal.output.append(line)

    def cleanup(self):
        if self.button:
            self.button.setEnabled(True)
        if self.spinner:
            self.spinner.setVisible(False)

        if hasattr(self, 'stop_button'):
            self.stop_button.setEnabled(False)


    def stop_command(self):
        if hasattr(self, 'worker') and self.worker:
            self.worker.stop()

