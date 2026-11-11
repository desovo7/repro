# WebShop R2 aligned to ALFWorld R2

Run: `gradus-webshop-h800-s0-r2aligned-20261109`.
Model: `Qwen/Qwen2.5-1.5B-Instruct`; seed: 0; training: 150 iterations.

[`webshop/adapter.py`](webshop/adapter.py) extracts WebShop-visible states,
[`webshop/r2_credit.py`](webshop/r2_credit.py) implements the transferred ALFWorld
R2 credit computation, and [`webshop/core.py`](webshop/core.py) manages the online
engine. Supporting modules preserve collector/trainer integration, actor checks,
environment/driver state, and full checkpoint retention.

All nine runtime modules match the run's `webshop_release.files` manifest.
The transfer was checked against ALFWorld R2 over 150 rounds, 19,200 trajectories,
and 300 tensor-injection comparisons. Historical execution checks are summarized
in [results.json](../../reports/gradus/results.json).

Run CPU tests from this directory:

```bash
python -m unittest webshop.test_webshop webshop.test_r2_aligned -q
```

NumPy is required; the tensor-path test also requires PyTorch. Training enters
through `python -m webshop.entry CONFIG.json`. Original configuration and
launch/environment/monitor scripts are in [`frozen_run/`](frozen_run/).
The prepared layout is `RUN/{webshop,source,data,config.json,launch.sh,environment.sh,monitor.py}`;
`source` contains the pinned native trainer. The original environment also put
`deps` and `deps-direct` dependency directories on `PYTHONPATH`.

For a new run, install dependencies and benchmark assets, prepare a new directory,
adapt absolute paths and identities, and recompute configuration and code/data
hashes. `webshop/entry.py` verifies these and rejects stale manifests. The scripts
preserve the H800/systemd environment, including the 128 MiB per-process JVM cap.

Retention protects the top three complete validation checkpoints and the latest
complete resume checkpoint, plus diagnostic iterations 115 and 135. The completed
run's best checkpoints were 150, 145, and 120, with latest 150. Large checkpoint
files and datasets are outside this source publication.

At iteration 150 the fixed development-panel success rate was 65.04%, versus
70.64% for local GiGPO. This single-seed experiment did not establish a performance
improvement. See [results](../../reports/gradus/results.json) and
[source provenance](../../snapshot_manifest.json).
