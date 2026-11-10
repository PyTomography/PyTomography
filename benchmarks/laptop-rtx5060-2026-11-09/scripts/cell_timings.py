"""Per-cell wall times of the executed tutorial notebooks, from the timestamps nbclient records in each cell.

    python cell_timings.py <notebooks folder> <out.json>

Writes {tutorial: [{"cell": index, "s": seconds, "code": first line of the cell}, ...]} for every code cell that ran,
in notebook order. Cells after a failure keep the outputs the notebook was saved with, and no timing, so they are left
out.
"""
import datetime as dt
import json
import sys
from pathlib import Path


def seconds(meta: dict) -> float | None:
    start, end = meta.get("iopub.execute_input"), meta.get("shell.execute_reply")
    if not (start and end):
        return None
    t = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00"))  # noqa: E731
    return round((t(end) - t(start)).total_seconds(), 1)


folder, out = Path(sys.argv[1]), Path(sys.argv[2])
timings = {}
for path in sorted(folder.glob("*.ipynb")):
    cells = json.loads(path.read_text(encoding="utf8"))["cells"]
    rows = []
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        s = seconds(cell.get("metadata", {}).get("execution", {}))
        if s is not None:
            first = "".join(cell["source"]).strip().split("\n")[0][:100]
            rows.append({"cell": i, "s": s, "code": first})
    timings[path.stem] = rows
out.write_text(json.dumps(timings, indent=1) + "\n", encoding="utf8")
print(f"{len(timings)} notebooks, {sum(len(r) for r in timings.values())} timed cells -> {out}")
