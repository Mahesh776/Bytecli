import json
from typing import Any


class JsonFormatter:
    def __call__(self, record: dict[str, Any]) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": record.get("time", ""),
            "level": record.get("level", ""),
            "module": record.get("name", ""),
            "function": record.get("function", ""),
            "line": record.get("line", 0),
            "message": record.get("message", ""),
        }
        if record.get("exception"):
            log_entry["exception"] = str(record["exception"])
        if record.get("extra"):
            log_entry["extra"] = record["extra"]
        return json.dumps(log_entry, default=str)
