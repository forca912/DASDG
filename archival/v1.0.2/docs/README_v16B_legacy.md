# v16B 独立测试实验包

## 目的

v16B 用固定的 6 个随机种子、2 个数据集、3 个全局缺失率和 3 个持续缺失长度，在独立测试集上比较：

- `graph_no_reliability_channel`：v15 已锁定的本文模型，直接读取 v15 的独立测试结果；
- `graph_wavenet_adaptive`：主要时空图基线；
- `gru_mask_duration`：时间序列基线；
- `persistence`、`history_mean`：非学习基线。

v16B 不允许依据测试集重新选择模型或超参数。外部模型只使用验证集选择最佳 epoch，随后才计算独立测试指标。

## 服务器上运行

先进入包目录，并确认数据目录实际存在。不要盲目使用默认路径：

```bash
cd /root/autodl-tmp/traffic_missing_forecasting_tools_v16b_primary
ls data/processed/PEMS04/complete.npz data/processed/PEMS08/complete.npz
```

如果 `data` 是软链接，检查它指向的目录：

```bash
readlink -f data
```

先做预检：

```bash
python preflight_v16b.py
```

预检必须显示 108 个掩码文件、两个数据集的 `complete.npz` 和 `cuda_available: true`。预检失败时不要启动长任务。

### 第一阶段：主要时空图基线

这是决定论文主终点的关键阶段：

```bash
nohup python -u run_v16b_primary.py \
  --models graph_wavenet_adaptive \
  --skip-naive \
  --output-root outputs/v16b_primary \
  > outputs/v16b_graph_wavenet.log 2>&1 &
echo $! > outputs/v16b_graph_wavenet.pid
```

查看进度：

```bash
tail -f outputs/v16b_graph_wavenet.log
```

确认进程和 GPU：

```bash
ps -p $(cat outputs/v16b_graph_wavenet.pid) -o pid,etime,cmd
nvidia-smi
```

### 第二阶段：时间序列基线和非学习基线

只有第一阶段完成并且日志最后出现 `V16B_PRIMARY_RUN_COMPLETE` 后再运行：

```bash
nohup python -u run_v16b_primary.py \
  --models gru_mask_duration \
  --skip-naive \
  --output-root outputs/v16b_primary \
  > outputs/v16b_gru.log 2>&1 &
echo $! > outputs/v16b_gru.pid
```

GRU 完成后，运行非学习基线：

```bash
python -u run_v16b_primary.py \
  --models graph_wavenet_adaptive \
  --output-root outputs/v16b_primary \
```

由于运行器会跳过已完成目录，上述命令不会重新训练 Graph WaveNet，但会补齐 `persistence` 和 `history_mean`。如果 Graph WaveNet 尚未完成，不要执行这一步。

## 结果回传与分析

把服务器目录中的以下文件下载到本地同一个文件夹：

```text
outputs/v16b_primary/external_horizon_metrics.csv
outputs/v16b_primary/naive_horizon_metrics.csv
```

同时保留每个模型目录中的 `config.json`、`efficiency.json`、`test_metrics.csv`、`validation_metrics.csv` 和 `best.pt`，用于完整性复核。下载后运行：

```bash
python analyze_v16b_primary.py \
  --v16b-root outputs/v16b_primary \
  --v15-summary D:/lizhuorui/论文/ccf c论文/fdr_v15_confirm_results_20260811/outputs/fdr_representation_v15_confirm/summary.csv
```

分析结果会写入 `outputs/v16b_primary/analysis/`。若 `integrity_audit.json` 的 `overall_ok` 不是 `true`，不能进入论文结果撰写。

## CCF-C 写作门槛

只有同时满足以下条件，才开始撰写包含实验结果的论文：

1. 108 个条件的 Graph WaveNet、GRU、Persistence 和 Historical Mean 结果完整；
2. 每个学习模型均有独立测试集指标，且模型选择只基于验证集；
3. 本文模型相对 Graph WaveNet 的 6 个种子平均相对 MAE 小于 0，至少 5/6 个种子方向支持，双侧精确符号翻转检验 `p < 0.05`；
4. PEMS04 与 PEMS08 分层方向均不为正；
5. 3/6/9/12 步误差、缺失率分层、持续长度分层、参数量和推理时延均已核验；
6. 任一条件不满足时，只写方法、数据和实验协议，不写“总体优于基线”的结论。

v16A 目前只完成了验证集基线锁定，不能替代 v16B 的独立测试。
