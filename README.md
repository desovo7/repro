# RePro / GRADUS experiment snapshots

This branch archives three completed GRADUS experiments and one continuation
with **Qwen/Qwen2.5-1.5B-Instruct**, training seed **0**. The original three runs
each trained **150 iterations**; the new polarity intervention resumes aligned
iteration **115** for **35 additional iterations**, ending at **150**.
Each experiment has its own import root.

**[当前实验情况与问题（中文，2026-11-11）](reports/gradus/EXPERIMENT_STATUS.md)**
summarizes ALFWorld and all WebShop results, evaluation protocols, known
problems, completed checks, checkpoint status, and proposed next validations.

| Snapshot | Run | Main implementation |
| --- | --- | --- |
| [ALFWorld R2](experiments/alfworld_r2/README.md) | `gradus-qwen25-h800-s0-r2` | `repro_v83/gradus.py` |
| [WebShop R2](experiments/webshop_r2/README.md) | `gradus-webshop-h800-s0-r2-jvmcap-20261108` | `webshop/core.py` |
| [WebShop R2 aligned](experiments/webshop_r2_aligned/README.md) | `gradus-webshop-h800-s0-r2aligned-20261109` | `webshop/adapter.py`, `r2_credit.py`, `core.py` |
| [WebShop R2 polarity intervention](experiments/webshop_r2_polarity/README.md) | `gradus-webshop-h800-s0-polarity-from115-20261111` | Three independent credit gates, jointly enabled for the 35-round continuation |

ALFWorld code comes from the frozen `gradus-20261106a` release. All WebShop
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

Run tests from the individual experiment directory; all WebShop snapshots use
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
| WebShop R2 polarity, aligned115→150 | **73.37%** | **0.8555** |

The original aligned version did not outperform local GiGPO. The new combined
intervention achieved **1127/1536** successes, **+2.73 percentage points** over
GiGPO (paired task-cluster 95% interval **[+0.26, +5.27]**). It did not meet every
behavior target: **67 format-invalid actions (0.75%)** and **one option-loop
episode** remain; unexecutable actions increased from **187 to 269** relative to
the aligned version. Timeout rate was **1.89%**. This is one training seed, three
joint interventions, and an already used development panel, so component effects
and training-seed robustness remain unestablished.

See the [full continuation report](reports/gradus/evidence/polarity_20261111/README.md).
All **4608 per-episode metric rows** for the candidate and two reused references
are published. Recompute their counts, means, and paired intervals with NumPy:

```bash
python3 scripts/verify_polarity_results.py
```

This recount uses extracted metrics, not the unpublished raw observation text.
Numeric summaries and historical
execution checks are in [results.json](reports/gradus/results.json).
Publication-time checks are recorded separately in
[publication_checks.json](reports/gradus/publication_checks.json).
