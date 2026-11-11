# RePro / GRADUS experiment snapshots

This branch archives three completed GRADUS experiments with
**Qwen/Qwen2.5-1.5B-Instruct**, training seed **0**, and **150 iterations**.
Each experiment has its own import root.

| Snapshot | Run | Main implementation |
| --- | --- | --- |
| [ALFWorld R2](experiments/alfworld_r2/README.md) | `gradus-qwen25-h800-s0-r2` | `repro_v83/gradus.py` |
| [WebShop R2](experiments/webshop_r2/README.md) | `gradus-webshop-h800-s0-r2-jvmcap-20261108` | `webshop/core.py` |
| [WebShop R2 aligned](experiments/webshop_r2_aligned/README.md) | `gradus-webshop-h800-s0-r2aligned-20261109` | `webshop/adapter.py`, `r2_credit.py`, `core.py` |

ALFWorld code comes from the frozen `gradus-20261106a` release. Both WebShop
implementations come from their actual run directories and match the SHA-256
hashes recorded by those runs. Historical configuration and launch files retain
their original machine paths.

## Source and environment

The native trainer and environments are supplied by `vendor/verl_agent`, pinned
to upstream commit `20bd331bdbc9026a5668e11362178e10ab7400c8`.
All 1,334 tracked upstream files were compared with the native source used by
each run: all matched. See [upstream equivalence](reports/gradus/upstream_equivalence.json).

```bash
git clone --recurse-submodules --branch codex/gradus-alfworld-webshop-20261110 https://github.com/desovo7/repro.git
cd repro
python3 scripts/verify_snapshots.py --with-submodule
```

For an existing checkout, run `git submodule update --init --recursive` first.
`python3 scripts/verify_snapshots.py` checks published files without the submodule
or training dependencies. Installed versions observed in the training environment
at publication time are recorded in [runtime metadata](reports/gradus/runtime_metadata.json).
This inventory is not a complete dependency lockfile. WebShop additionally uses
Java, its product data/search index, and the dependencies described by upstream.

## Reusing the code

Run tests from the individual experiment directory; both WebShop snapshots use
the package name `webshop`. Their READMEs list the commands. Training requires
separately installed dependencies, model weights, benchmark data, two GPUs, and
configuration paths adapted to the new machine. WebShop verifies code, data, and
configuration identities; a new run must rebuild these after changing paths.
The archived launch scripts document the original run layout and are not
portable installers.

This publication includes source, tests, original configurations, compact results,
and provenance. Large model/checkpoint files, downloaded datasets, and full
training trajectories remain outside Git. The small ALFWorld trace fixture is
included because the archived tests use it.

## Recorded results

The fixed WebShop development panel contains 512 task instances with three
generation repeats. Iteration 150 results are:

| Model | Success rate | Task score |
| --- | ---: | ---: |
| Local GiGPO | 70.64% | 0.8399 |
| WebShop R2 | 65.43% | 0.7576 |
| WebShop R2 aligned | 65.04% | 0.7922 |

The aligned version did not outperform local GiGPO. These are single-training-seed
results on an already used development panel. Numeric summaries and historical
execution checks are in [results.json](reports/gradus/results.json).
Publication-time checks are recorded separately in
[publication_checks.json](reports/gradus/publication_checks.json).
