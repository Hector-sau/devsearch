# Clean-environment rehearsal of the documented startup procedure (CPU DIAGNOSTIC), 2026-10-06

**CPU diagnostic rehearsal, not a GPU run.** Fresh clone of the private repository branch over HTTPS, a fresh runtime root under `/workspace`, the
documented commands of `docs/GPU_QUICKSTART.md` (CPU rehearsal block), the shipped `configs/gpu-cpu-rehearsal.toml`, a random tiny stand-in encoder with the
reference vocabulary, and the real data rebuilt from `pytorch/tutorials@11512db7`. The interpreter was a clean Python 3.12 environment installed from the
hashed CUDA lock (CPU use). Judgments were SCRIPTED (`scripted-oracle`), never a review. Nothing here says anything about a GPU, MiniLM or ranking quality.

What the rehearsal found and fixed before this final run (all in the repository now): manual `devsearch.*` commands could run an older installed wheel
(`gpu.py cmd` added); a surviving campaign process after a killed `gpu.py` was reported as an anonymous failed step (`PREREQUISITE campaign-lock`);
a rehearsal user had no way to produce judgments (`gpu.py rehearsal-judge`, refused on accelerator profiles); private-repository access and the input
checklist were undocumented (quickstart sections 2 and 4). Earlier runs of the same procedure also showed that a `SIGKILL` of `prepare` during the
tokenizer comparison and during the data build recovers on rerun, and that a surviving campaign refuses a second owner.

Final transcript (commit below; paths are under `/workspace`; exit codes are `gpu.py`'s):

```text
commit 0aeb6ed19bce5bea503dbb08a53ec4ac0d896a9a

### standin-model
exit 0

### check
exit 0

### prepare (no judgments)
2026-10-06T10:48:35Z data: rebuilt with the pinned tokenizer; corpus corpus-4367f67345460c9769cdf9b373dd3982, 197 test questions, 9852 review items; identical to the frozen dataset
PREREQUISITE review-bank: no review judgments yet: judge every batch in /workspace/rehearsal-runtime/review/batches/A and /workspace/rehearsal-runtime/review/batches/B (two independent primary passes;
exit 3

### run before prepare completes
PREREQUISITE prepare: run `gpu.py prepare` until it completes (environment, receipts, model, data, review bank, bundle)
exit 3

### import ONE scripted pass (A) by hand, then prepare
1
PREREQUISITE review-bank: review coverage incomplete: 0 of 9852 items final, states {'needs primary reviews (1 of 2)': 9852}. Sealing and formal evaluation stay blocked. Next: judge every batch in /wo
exit 3
initial seal: absent; bundle manifest: absent

### run with one pass of two
PREREQUISITE prepare: run `gpu.py prepare` until it completes (environment, receipts, model, data, review bank, bundle)
exit 3

### rehearsal-judge (pass B imported; A already present)
2026-10-06T10:48:39Z rehearsal-judge: two scripted passes imported (diagnostic; not a review). Next: gpu.py prepare
exit 0

### prepare
2026-10-06T10:48:45Z prepare: complete; bundle 518ad46e68af (diagnostic: True). Next: gpu.py diagnose
exit 0

### run before diagnose
PREREQUISITE diagnose: run `gpu.py diagnose` first: the small real-model training, inference and recovery check must pass for this code, mod
exit 3

### diagnose
2026-10-06T10:49:48Z diagnose: passed on cpu-diagnostic with the pinned model (crash at mid_stage:train:full:42, resumed, status complete). Next: gpu.py run
exit 0
{'passed': True, 'crash_exit': 137, 'status_exit': 0, 'train_full_attempts': 2, 'fault': 'mid_stage:train:full:42'}

### run, then HARD KILL of everything (machine-crash simulation) after training starts
stages at kill: baseline_validation data_check preflight train:frozen:42 
/workspace/rehearsal.sh: line 36: 13279 Killed                  setsid $PYML scripts/gpu.py run --config $C > /workspace/rehearsal-run1.out 2>&1
stale lock left: yes; leftover processes: 0

### SECOND run after the crash (resume)
exit 0
2026-10-06T10:53:28Z run: complete and every artifact re-verified. Outputs under /workspace/rehearsal-runtime/campaign/run (final report, release, backup receipt); backup in /workspace/rehearsal-runtime/backup. cpu-diagn

### status
done: read the final report under <root>/campaign/run and copy the backup off the server
campaign_status_exit 0
final reports: 2; backup entries: 1
```

Confirmed: nothing is sealed and no bundle exists while review coverage is incomplete (one pass of two: `0 of 9852 items final`, no `initial-seal.json`,
no `bundle-manifest.json`), and `run` is refused (exit 3) until `prepare` completes, again until `diagnose` has passed, and the campaign resumed after a
hard kill of every process (stale lock taken over) to `complete and every artifact re-verified`; both campaign reports are stamped diagnostic.
