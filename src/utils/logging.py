import json
import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


class ExtraJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        reserved = {
            "name", "msg", "args", "levelname", "levelno",
            "pathname", "filename", "module", "exc_info",
            "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process",
        }

        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in reserved
        }

        if extras:
            log["extra"] = extras

        if record.exc_info:
            log["exception"] = self.formatException(record.exc_info)

        return json.dumps(log, ensure_ascii=False)


def setup_logging(
    level: int = logging.INFO,
    log_dir: str = "logs",
    log_name: str = "app.log",
) -> None:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    formatter = ExtraJsonFormatter()

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    file = TimedRotatingFileHandler(
        filename=str(log_path / log_name),
        when="midnight",
        backupCount=14,
        encoding="utf-8",
    )
    file.setFormatter(formatter)

    root.addHandler(console)
    root.addHandler(file)