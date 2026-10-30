# RePro v8.3.1 — Tier1（1A＋1B）

本分支保存 **Tier1（1A＋1B）** 的代码与冻结配置，对应 `codex/repro-v831-tier1`。
两版共用已部署的核心源码 `0607ff62e225`，通过算法开关区分。`configs/default.yaml` 是本分支的默认入口配置；其配套 contract/card 均保留原始字节。

| 分支 | Tier1A | Tier1B | Tier2 | 默认配置来源 |
|---|---|---|---|---|
| [Tier1](https://github.com/desovo7/repro/tree/codex/repro-v831-tier1) | 开 | 开 | 关 | A800 Tier1B 阶段 |
| [Tier1＋Tier2](https://github.com/desovo7/repro/tree/codex/repro-v831-tier1-tier2) | 开 | 开 | 开 | H800 Tier2 阶段 |

## 方法与配置

- Tier1A：以 `(norm(obs), p, b_nav)` 解混叠非全败组的 GiGPO 锚点；锚点过稀时回退。
- Tier1B：对全败组计算规则势函数差分、No-Op 截断 GAE、同题同预算桶中心化，并按 AF 回答 token 的标准差缩放后注入。
- Tier2：使用前 4 轮已提交训练的非全败组，按轨迹首次访问统计 96 格商空间，以 `alpha0=5` 收缩并投影。成功终止值为 1，失败终止值为 0；当前轮和评估轨迹不进入当轮表。
- 两版均为 Qwen2.5-1.5B-Instruct、ALFWorld、GiGPO；150 轮、16×8 条轨迹/轮、50 步、2048/512 token、双卡 TP2。所有训练参数保留原配置。

完整规范见 [原始 v8.3.1 规范](reports/v83_20261030/implementation_spec.v831.original.md)；原文中的引用/算术勘误见 [errata](reports/v83_20261030/implementation_errata.v831.json)，保证范围与统计解释见 [实现核对记录](reports/v83_20261030/method_summary_consistency_20261030.json)。

表值是带符号的收缩势函数；96 格有有限估值不代表均有经验样本。原始 TD 的符号不等于最终 Actor 优势符号。AF 不保证原生优势为零，因为原生配置包含非法动作惩罚。代码快照不代表完整阶段训练或收益验证已经完成。

## 获取与校验

```sh
git clone --branch codex/repro-v831-tier1 --recurse-submodules https://github.com/desovo7/repro.git
cd repro
python scripts/verify_snapshot.py
```

上游 [verl-agent](https://github.com/langfengQ/verl-agent) 作为 `vendor/verl_agent` 子模块，固定到 `20bd331bdbc9026a5668e11362178e10ab7400c8`，保留其原始代码、许可证和安装说明。现有克隆可执行 `git submodule update --init --recursive` 获取。

CPU 规则与历史表检查需要 Python、NumPy；评估合约检查还需 PyYAML。以下检查不运行 GPU 训练：

```sh
python -m unittest test_repro_v83_rules test_repro_v831_table test_repro_v831_runtime test_v831_eval_contract
```

原生 collector/训练入口检查还需要上游运行环境及 ALFWorld 数据，入口为 `ops_v83/cpu_gate.py`。GPU 运行依赖及环境安装按固定版本上游说明准备。

## 代码与实验文件

- `repro_v83/`：状态追踪、锚点、信用、价值表以及原生训练接入。
- `repro_external/`、`repro_integration/`、`repro_core/`：冻结版本携带的接入与兼容依赖；旧 RePro 分支由配置 `repro.enabled=false` 关闭。
- `tests_v83/`、`test_repro_v83*.py`：规则、数值、历史隔离、断点与原生接入检查。
- `ops_v83/`：源码冻结、数据准备、阶段配置与启动检查。
- `ops_v831_eval/`：固定 step150 的 140 seen＋134 unseen 评估及独立重数；见其 README。
- `reports/v83_20261030/prepared_configs_r6/`：原始五套阶段配置、合约与实验卡。分支默认来源是 `repro-v831-qwen25-tier1b-a800-s0-r6`。
- `reports/v83_20261030/replayed_traces.jsonl`：真实轨迹单测所需的 45 条重放样例；训练全量日志、权重和完整数据集不包含在快照中。
- `snapshot_manifest.json`：本分支文件摘要；`release_0607ff62e225.json` 保留原始 58 文件摘要。

## 训练入口与阶段顺序

训练入口是 `python -m repro_v83.entrypoint CONFIG.yaml`，正式部署使用 `ops_v83/launch.py` 的合约和前置阶段检查。
`configs/default.yaml` 保留原服务器的模型、数据、输出、PYTHONPATH 绝对路径。它是可追溯的原始部署配置；移植到新机器时须重新生成本机配置与合约，并完成原生 CPU/数据检查，再启动训练。

A800 顺序：1A → 1A＋1B。H800 顺序：1A → 1A＋1B → 1A＋1B＋2。
每阶段从同一原始模型重新训练完整 150 轮，并在同机完成前一阶段检查点及全量评估后推进。这里保留原有阶段门槛；分支名称表示算法组合，不表示该阶段已经运行完成。
