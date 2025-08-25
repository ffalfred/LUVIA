
from PyQt6.QtCore import QObject, QThread, pyqtSignal
import subprocess
import logging
from datetime import datetime

class CommandWorker(QObject):
    finished = pyqtSignal(str, str, str)  # command, output, error

    def __init__(self, command):
        super().__init__()
        self.command = command

    def run(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            result = subprocess.run(self.command, shell=True, capture_output=True, text=True)
            output = result.stdout.strip()
            error = result.stderr.strip()
            self.finished.emit(self.command, output, error)
        except Exception as e:
            self.finished.emit(self.command, "", f"Exception: {str(e)}")

class CommandManager:
    def __init__(self, terminal=None, log_file="command_log.txt"):
        self.terminal = terminal
        self.log_file = log_file

        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )

    def execute_command(self, command):
        self.thread = QThread()
        self.worker = CommandWorker(command)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.handle_result)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def handle_result(self, command, output, error):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"{timestamp} - Executed: {command}")
        if output:
            logging.info(f"{timestamp} - Output: {output}")
        if error:
            logging.error(f"{timestamp} - Error: {error}")

        if self.terminal:
            self.terminal.output.append(f"> {command}\n")
            if output:
                self.terminal.output.append(output + "\n")
            if error:
                self.terminal.output.append("Error:\n" + error + "\n")
