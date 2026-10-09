"""Stand-in probe workers for run_worker tests: behavior is selected by the spec's `mode`."""

import json
import signal
import sys
import time
from pathlib import Path

spec = json.loads(Path(sys.argv[1]).read_text())
mode = spec.get("mode")
if mode == "sleep":
    time.sleep(60)
elif mode == "ignore_term":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(60)
elif mode == "crash":
    sys.stderr.write("simulated driver crash\n")
    raise SystemExit(3)
elif mode == "ok":
    Path(sys.argv[2]).write_text(json.dumps({"ok": True, "echo": spec.get("value")}))
