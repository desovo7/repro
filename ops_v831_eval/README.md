# v8.3.1 固定 step150 评估

本目录是未来阶段评估的准备工具。未加载训练 hook；评估不向 Tier2 历史写入数据。
每个阶段在其训练主机上单独建立目录，使用同一冻结任务清单和双卡 TP2。

## 执行顺序

以下命令中的路径应替换为该主机上的实际绝对路径。准备步骤只查询 GPU 身份，不分配 GPU；合并与完整张量核验隐藏 CUDA。

```sh
python ops_v831_eval/prepare.py \
  --config <host-root>/configs/<run>.yaml \
  --protocol <evaluation_protocol.json> \
  --output <host-root>/evaluations/<run>-step150 \
  --gpu-indices 2 3 --ray-temp /tmp/rv831-e150-a

python <eval-root>/merge.py --root <eval-root>
python <eval-root>/run.py --root <eval-root> --preflight
python <eval-root>/run.py --root <eval-root>
python <eval-root>/verify_results.py <eval-root>
```

运行使用已安装官方训练依赖的 Python。正式评估可由主线程放入持久服务；`run.py` 设置冻结的环境变量并记录本次预留双卡时间。使用数字 GPU 编号与 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，启动时检查两张卡空闲且 GPU UUID/机型匹配。持续共享资源管理由外层服务负责。

## 输入与输出

- 原训练配置、card、contract 必须一致；完成检查点必须为本 run 的 `global_step_150/V83_COMPLETE.json`。runtime/RNG 摘要、阶段及 Tier2 历史均核验。
- 捕获两个原生模型 rank 文件、模型配置和 tokenizer；不复制 optimizer。使用同一官方 `scripts/model_merger.py merge/test`，全模型张量核验通过才生成 `merge_verification.json`。
- 必须使用预训练冻结协议原件（SHA256 `b948997c823517103178d391d7e0d745bbc341d0565733042341f0c5b62960c3`）。将协议内相对路径映射到本机 `v83_data/alfworld`；保留全部 140 seen + 134 unseen 的 ID、顺序、文件摘要、分项和种子。冻结源文件、模型、全部数据与规则摘要，所有主机均执行校验。
- `verify_results.py` 独立重数十个 batch 文件，验证每题恰好一次、UUID 无重复、末步 won 与奖励一致、分项指标一致，然后写 `phase_evaluation.json`。
- 阶段 summary 包含 `status=complete`、训练 `run_id`、`checkpoint_iteration=150`、`seen_games=140`、`unseen_games=134`，以及配置、模型、协议、任务、完成标记、轨迹和核验记录摘要，兼容现有阶段 gate。

## 约束

固定 32 环境、50 步、2048/512、temperature 0.4、top_p 1、top_k -1、模型 seed0、每题一次；无 best-of 或失败题重试。`EvaluationRollout.world_size=1` 表示一个独立引擎，内部 TP 按 card 固定为 2。

评估目录不覆盖已有结果。首次启动即创建独占尝试记录，即使首批尚未写出也禁止在同目录重试。运行中断后保留失败尝试及成本，另建目录并记录新尝试；当前工具不支持保留随机采样状态的中途续跑。单 seed、每题一次且任务已用于此前诊断，不据此宣称盲测或多 seed 优势。

评估主流程从 `evaluation/repro_step150_full_20261030` 的已用 evaluator、任务分配器和独立 recount 复制并适配；历史文件保持不变。

本次 12 项 CPU 合约测试使用合成文件和轨迹，只验证完整性、口径、错误拒绝和摘要接口；不是实际模型合并或 GPU 评估的证据。完整 prepare 测试仅将本机缺少的 OmegaConf 解析依赖替换为等价的 JSON 输入读取，并模拟 GPU 身份查询；真实服务器上的安装依赖、完成检查点、合并与执行仍须逐项实际核验。
