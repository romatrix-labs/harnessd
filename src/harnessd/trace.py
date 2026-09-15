import json
import time
from pathlib import Path
from typing import Any


class TraceLogger:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.events: list[dict[str, Any]] = []

    def log(self, event: str, **data: Any) -> None:
        record = {"ts": time.time(), "event": event, **data}
        self.events.append(record)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
