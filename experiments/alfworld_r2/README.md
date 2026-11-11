# ALFWorld R2

Run: `gradus-qwen25-h800-s0-r2`; frozen release: `gradus-20261106a`.
Model: `Qwen/Qwen2.5-1.5B-Instruct`; seed: 0; training: 150 iterations.

The GRADUS credit implementation is [`repro_v83/gradus.py`](repro_v83/gradus.py).
Collection, trace parsing, actor injection, and trainer integration are preserved
in `repro_v83`, with supporting `repro_external`, `repro_core`, and
`repro_integration` modules. Historical package names are retained to keep runtime
source byte-identical to the experiment.

[`config.frozen.yaml`](config.frozen.yaml) is the exact run configuration.
The entrypoint is `python -m repro_v83.entrypoint CONFIG.yaml`, with this directory,
the pinned `vendor/verl_agent`, and its ALFWorld package on `PYTHONPATH`.
The configuration includes original H800 model, data, output, Ray, and executable
paths. Copy it to a new run and update paths and identities before use.
It records historical checkpoint settings; future runs must retain the best
three complete validation checkpoints and the latest complete resume checkpoint.

Run CPU regressions from this directory:

```bash
python -m unittest discover -s tests_v83 -q
python -m unittest discover -s . -p 'test_repro*.py' -q
```

The full suites require NumPy and PyTorch. Native integration tests additionally
require the pinned trainer and its dependencies. The fixture under `reports/`
is used by the archived tests. See [source provenance](../../snapshot_manifest.json).
