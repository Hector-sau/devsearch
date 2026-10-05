"""Execution controller: a small sequential campaign runner (docs/10 §2, §5).

Layout: `lock` (single-owner leases), `records` (plan/state/stage-manifest/event formats and atomic
writers), `hostfile` (host binding checks), `bundle` (the prepared inputs), `doctor`/`probe` (preflight and
calibration), `plan` (the immutable resolved plan), `stages`/`controller` (the stage graph, recovery and
exit codes), `acceptance` (gate evidence) and `backup`. It calls existing module APIs; nothing here
re-implements training, scoring or search.
"""
