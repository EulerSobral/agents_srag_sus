import os
import sys
import logging
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS_DIR = os.path.join(PROJECT_ROOT, "output", "logs")

def setup_logger(log_level: int = logging.INFO) -> logging.Logger:
    """
    Configura o sistema de logging centralizado para governança e auditoria.
    Cria handlers para Console (stdout) e Arquivo Persistente em output/logs/agent_execution.log.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file_path = os.path.join(LOGS_DIR, "agent_execution.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Evita duplicação de handlers se a função for chamada mais de uma vez
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file_path) for h in root_logger.handlers):
        formatter = logging.Formatter(
            fmt="%(asctime)s - [%(levelname)s] - [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 1. Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # 2. Persistent File Handler
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        logging.info(f"=== Governança & Logging Inicializado | Log File: {log_file_path} ===")

    return root_logger
