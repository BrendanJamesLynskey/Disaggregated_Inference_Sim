"""Chrome trace-event export: open the JSON in https://ui.perfetto.dev or chrome://tracing.

One "process" row per instance (plus the KV link); each batch step becomes a
complete (``ph: "X"``) slice, and the in-system population is a counter track.
"""

from __future__ import annotations

import json
from pathlib import Path


class Tracer:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.events: list[dict] = []
        self._pids: dict[str, int] = {}

    def _pid(self, name: str) -> int:
        if name not in self._pids:
            pid = len(self._pids) + 1
            self._pids[name] = pid
            self.events.append({"ph": "M", "name": "process_name", "pid": pid, "tid": 0,
                                "args": {"name": name}})
        return self._pids[name]

    def span(self, where: str, cat: str, name: str, start: float, dur: float) -> None:
        if self.enabled:
            self.events.append({"ph": "X", "name": name, "cat": cat, "pid": self._pid(where),
                                "tid": 0, "ts": start * 1e6, "dur": dur * 1e6})

    def counter(self, name: str, values: dict, t: float) -> None:
        if self.enabled:
            self.events.append({"ph": "C", "name": name, "pid": self._pid("counters"),
                                "ts": t * 1e6, "args": values})

    def export(self) -> dict:
        return {"traceEvents": self.events, "displayTimeUnit": "ms"}


def write_trace(trace: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(trace))
