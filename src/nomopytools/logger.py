import logging
from pathlib import Path


class Logger:
    class InfoFilter(logging.Filter):
        def filter(self, record):
            """Filtering INFO (20) and WARN (30) logs."""
            return record.levelno in [20, 30]

    @classmethod
    def get_logger(cls, dunder_name: str, logs_folder: str | Path):

        if isinstance(logs_folder, str):
            logs_folder = Path(logs_folder)

        formatter = logging.Formatter(
            "%(asctime)s::%(name)s::%(levelname)s::%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        logger = logging.getLogger(dunder_name)
        logger.setLevel(logging.INFO)

        info_file_handler = logging.FileHandler(logs_folder / "info.log")
        info_file_handler.setFormatter(formatter)
        info_file_handler.addFilter(cls.InfoFilter())
        info_file_handler.setLevel(logging.INFO)

        error_file_handler = logging.FileHandler(logs_folder / "error.log")
        error_file_handler.setFormatter(formatter)
        error_file_handler.setLevel(logging.ERROR)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        logger.addHandler(info_file_handler)
        logger.addHandler(error_file_handler)
        logger.addHandler(stream_handler)

        return logger
