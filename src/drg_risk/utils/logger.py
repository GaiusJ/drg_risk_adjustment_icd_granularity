import logging
import os
from datetime import datetime
import sys
from drg_risk import PROJECT_ROOT

class Logger:
    """
    Klasse zum einfachen dokumentieren und aufzeichnen von Daten über einen längeren Zeitraum.
    Verwendet eine automatische Verknüpfung der Systemzeit mit den Nachrichten zum besseren Verständnis der Protokolle.
    """
    def __init__(self, name, level = logging.INFO):
        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.log_dir = os.path.join(PROJECT_ROOT, "Logs")
        os.makedirs(self.log_dir, exist_ok = True)
        self.log_file = os.path.join(self.log_dir, f"log_{name}_{self.start_time}.log")

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        formatter = logging.Formatter('%(asctime)s - %(message)s')

        # File Handler
        try:
            file_handler = logging.FileHandler(self.log_file, mode = "w")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            print(f"FEHLER: Beim Logging ist der folgende Fehler aufgetreten: {e}")

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    # Funktionen, mit denen der Logger aufgerufen wird.
    def info(self, message): self.logger.info(message)
    def error(self, message): self.logger.error(message)
    def warn(self, message): self.logger.warning(message)