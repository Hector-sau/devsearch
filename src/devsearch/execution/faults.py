"""Crash-point injection for the failure rehearsal (docs/11 §4). Inert unless explicitly enabled.

Setting `DEVSEARCH_FAULT=<point>:<stage_id>` (or `<point>:*`) makes the process exit immediately with status
137 when it reaches that point, running no cleanup handlers, exactly like a killed process. Production runs
never set it; the rehearsal uses it to prove each documented crash window recovers correctly.

Points: `stage_outputs_written` (outputs exist, manifest not committed), `stage_manifest_committed`
(manifest committed, run-state not yet updated), `state_updated` (everything committed), `mid_stage`
(inside a stage's own work, e.g. between training epochs).
"""

import os
import sys

ENV = "DEVSEARCH_FAULT"
POINTS = ("stage_outputs_written", "stage_manifest_committed", "state_updated", "mid_stage")


def point(name: str, stage_id: str) -> None:
    configured = os.environ.get(ENV)
    if not configured:
        return
    wanted_point, _, wanted_stage = configured.partition(":")
    if wanted_point == name and wanted_stage in (stage_id, "*"):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(137)
