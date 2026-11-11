# WebShop R2

Run: `gradus-webshop-h800-s0-r2-jvmcap-20261108`.
Model: `Qwen/Qwen2.5-1.5B-Instruct`; seed: 0; training: 150 iterations.

This is the first completed WebShop R2 implementation. State parsing and the
credit engine are in [`webshop/core.py`](webshop/core.py); collector/trainer hooks
are in `integration.py`, actor checks in `actor_audit.py`, and full resume state
and checkpoint retention in `persistence.py` and `native_retention.py`.
Its credit implementation is separate from the later ALFWorld-aligned version.

The seven runtime modules match the run's `webshop_release.files` manifest.
[`frozen_run/config.json`](frozen_run/config.json), `launch.sh`, `environment.sh`,
and `monitor.py` preserve original settings and scripts.

Run CPU tests from this directory:

```bash
python -m unittest webshop.test_webshop -q
```

NumPy is required; the tensor-path test also requires PyTorch. Training enters
through `python -m webshop.entry CONFIG.json`. The prepared runtime layout is
`RUN/{webshop,source,data,config.json,launch.sh,environment.sh,monitor.py}`, with
`source` containing the pinned native trainer. The original run also used
`deps` and `deps-direct` dependency directories on `PYTHONPATH`.

For a new run, install dependencies and benchmark assets, prepare a new directory,
adapt absolute paths and identities, and recompute the configuration identity
hash and code/data SHA-256 manifests. `webshop/entry.py` defines these checks and
rejects stale manifests. The shell scripts are historical H800/systemd launch
scripts, not portable installers. The JVM heap was capped at 128 MiB per process.

Retention protects the top three complete validation checkpoints and the latest
complete checkpoint. Large checkpoints and datasets are outside this publication.
See [results](../../reports/gradus/results.json) and [hashes](../../snapshot_manifest.json).
