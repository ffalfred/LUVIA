
import os
import signal
import subprocess
import logging
from datetime import datetime
from PyQt6.QtCore import QObject, QThread, pyqtSignal

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

        try:
            self.process = subprocess.Popen(
                self.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid  # Start in a new process group
            )

            for line in self.process.stdout:
                if self._should_stop:
                    break
                line = line.strip()
                if line:
                    logging.info(line)
                    self.output_line.emit(line)

            self.process.wait()
        except Exception as e:
            self.output_line.emit(f"Error executing command: {e}")
        finally:
            self.finished.emit()

    def stop(self):
        self._should_stop = True
        if self.process and self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.output_line.emit("Process terminated.")
            except Exception as e:
                self.output_line.emit(f"Failed to terminate process: {e}")
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.output_line.emit("Process forcefully killed.")
                except Exception as e2:
                    self.output_line.emit(f"Failed to kill process: {e2}")

class CommandManager:
    def __init__(self, terminal=None, log_file="command_log.txt", button=None, spinner=None, stop_button=None):
        self.terminal = terminal
        self.log_file = log_file
        self.button = button
        self.spinner = spinner
        self.stop_button = stop_button
        self.command_finished_callback = None

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
        if self.stop_button:
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
        if self.stop_button:
            self.stop_button.setEnabled(False)
        if self.command_finished_callback:
            self.command_finished_callback()


    def stop_command(self):
        if hasattr(self, 'worker') and self.worker:
            self.worker.stop()
