
import os
import signal
import psutil
import subprocess
import threading
import time
from PyQt6.QtCore import QThread, pyqtSignal

BACKEND_SCRIPT = "horde"

class BackendWorker(QThread):
    output_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, input_folder, output_folder):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.process = None
        self.running = True
        self.pgid = None

    def run(self):
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            self.process = subprocess.Popen(
                ["luvia", BACKEND_SCRIPT,
                 "--folder_streets", self.input_folder,
                 "--output", self.output_folder],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True,
                env=env,
                preexec_fn=os.setsid
            )

            self.pgid = os.getpgid(self.process.pid)

            def read_stream(stream, signal):
                while self.running:
                    line = stream.readline()
                    if not line:
                        break
                    signal.emit(line.strip())
                stream.close()

            stdout_thread = threading.Thread(target=read_stream, args=(self.process.stdout, self.output_signal))
            stderr_thread = threading.Thread(target=read_stream, args=(self.process.stderr, self.error_signal))

            stdout_thread.start()
            stderr_thread.start()

            self.process.wait()
            self.running = False

            stdout_thread.join()
            stderr_thread.join()

        except Exception as e:
            self.error_signal.emit(f"Error running process: {str(e)}")

        self.finished_signal.emit()

    def stop(self):
        self.running = False
        if self.process and self.process.poll() is None and self.pgid:
            try:
                self.output_signal.emit("Sending SIGTERM to backend process group...")
                os.killpg(self.pgid, signal.SIGTERM)
                time.sleep(2)

                parent = psutil.Process(self.process.pid)
                children = parent.children(recursive=True)
                gone, alive = psutil.wait_procs([parent] + children, timeout=5)

                if alive:
                    self.output_signal.emit("Processes still alive after SIGTERM. Sending SIGKILL...")
                    for p in alive:
                        self.output_signal.emit(f"Killing process {p.pid}")
                        p.kill()
                    psutil.wait_procs(alive, timeout=5)

                self.output_signal.emit("Backend process group terminated.")

            except Exception as e:
                self.error_signal.emit(f"Failed to stop process: {str(e)}")

        self.finished_signal.emit()
