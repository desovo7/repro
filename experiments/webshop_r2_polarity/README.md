# WebShop R2：辅助信用门控联合干预

运行：`gradus-webshop-h800-s0-polarity-from115-20261111`。
模型 `Qwen/Qwen2.5-1.5B-Instruct`，训练种子 0。
从原对齐版完整检查点 115 继续训练至 150，**新增 35 轮、4,480 条训练轨迹**。
这是晚期续训分支，不是另一个从头训练的 150 轮种子。

本快照基于原对齐版提交 `a865ba3965f6da15462f159f6f7404124c3e0f13`，
九个运行模块与真实训练冻结文件逐字节一致。三个开关在本次训练同时启用：

| 开关 | 训练值 | 全败组中的行为 |
|---|---|---|
| `gate_invalid_noop` | `true` | 格式异常或语义 noop 的辅助项为零，退出信用桶；两个观测标签分别保留 |
| `return_credit_mode` | `neutral` | 可执行 item/detail→home/results 返回的 delta/raw/centered/注入全部为零，退出桶并切断该处辅助 GAE 传播 |
| `neutral_terminal_buy` | `true` | 已终止的可执行 Buy Now 辅助项为零，退出信用桶 |

原生奖励、GiGPO 优势、状态提取和历史支持集保留原契约。全败表示组内没有完全成功，
不等于没有部分任务分数或买错商品类别。返回完全中性比仅截断负单步差分更强；
后者不能保证经 GAE 和中心化后的信用为零。关闭全部开关时，计算与冻结对齐版一致。

## 检查与复算

在本目录执行：

```bash
python test_intervention.py
```

测试包含原 37 项和新增 5 项；需要 NumPy。PyTorch 缺失时仅张量路径测试跳过，
不要把这种运行称为 42 项全部通过。真实训练前的 H800 CPU 测试为 42 项全部通过。
仓库根目录可执行以下命令，检查三组完整逐轨迹指标的计数、均值和配对区间：

```bash
python scripts/verify_snapshots.py
python scripts/verify_polarity_results.py
```

第二项需要 NumPy。它复算已核验的 4,608 行指标，不替代从原始终止页面提取分数。
完整原始记录曾独立复算，摘要、批次哈希和所有指标行均已发布；完整文本不在 Git 中。

## 训练与历史分析脚本

训练入口为 `python -m webshop.entry CONFIG.json`。
[`frozen_run/config.json`](frozen_run/config.json) 保存实际配置；
`analysis/prepare_h800.py` 记录从完整父检查点迁移契约和保护状态的步骤。
`analysis/replay_intervention.py` 支持 CPU 反事实回放；`analysis/panel/` 保存冻结面板的
推理、记录和核验实现。这些历史脚本保留原服务器布局，不是可直接运行的通用安装器。
它们需要外部原始轨迹、模型/数据、上游依赖和完整父检查点；更改路径后须重新冻结身份与哈希。

该运行验证了 116 完整保存、实际重载并完成 117 更新，以及最终 150 的完整状态。
保留验证成功率最佳三份 **150、145、140**，最新完整状态 **150**，诊断保护点 **115、116**。
同分优先较后轮次。所有受保护检查点均核验文件大小和 SHA-256，完整权重及优化器文件未上传。

## 结果

最终 150 在固定 512×3 开发面板上成功 **1,127/1,536＝73.37%**，
原对齐版为 65.04%，GiGPO 为 70.64%。相对 GiGPO 的配对成功率差为
**+2.73 个百分点，95% 区间 [+0.26, +5.27]**。
格式异常 67 次、0.75%，仍未达到 <50 和 <0.6%；option_loop 为 1，未达到零。
超时 1.89%，训练新续训段返回导航 48 次。不可执行动作由原对齐版 187 次增加到 269 次。

这些是单训练种子、三机制联合、重复使用开发面板的结果，不能识别单个机制的贡献，
也不能证明跨训练种子稳定性或独立未见任务泛化。

[完整结果与限制](../../reports/gradus/evidence/polarity_20261111/README.md) ·
[冻结实验卡](../../reports/gradus/evidence/polarity_20261111/EXPERIMENT_CARD.md) ·
[来源与校验](../../reports/gradus/evidence/polarity_20261111/source_index.json)
