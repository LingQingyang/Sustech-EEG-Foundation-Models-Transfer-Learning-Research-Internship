# LaBraM × M3CV Task Singular Vector Analysis — Clean 1.0

这是对原有多版本脚本的完整重写。活动代码不再拼接 “legacy/new” 结果，也不再调用旧版本 Python 文件；五个任务、两个 replicate、六条 spectrum、heat map、STI 和绘图均由一个包内的固定接口完成。

## 1. 代码边界

| 文件 | 唯一职责 |
|---|---|
| `labram_tsv/config.py` | 五任务、72 个矩阵、训练与分析常量的唯一来源 |
| `labram_tsv/io.py` | 原子写入、哈希、规范化 5×2 run discovery、resume signature |
| `labram_tsv/data.py` | M3CV H5 契约、全文件审计、RAM staging、batch loader |
| `labram_tsv/model.py` | LaBraM 构建、checkpoint 严格加载、72 矩阵替换、功能评估 |
| `labram_tsv/training.py` | 十次独立 full fine-tuning；每次从同一 `W0` 开始 |
| `labram_tsv/geometry.py` | 纯 NumPy rank-one operator 几何和数值自测 |
| `labram_tsv/spectra.py` | **只实现六条 spectra** |
| `labram_tsv/heatmap.py` | `ΔW/W0` 相对更新能量表 |
| `labram_tsv/sti.py` | paper-aligned STI 与 matched-random null |
| `labram_tsv/plotting.py` | 只读 CSV 重画全部图片，不做分析 |
| `labram_tsv/run.py` | 唯一命令行入口 |

`spectra.py` 中严格只有以下六条科学结果：

1. Individual Energy
2. Individual Functional
3. Shared Energy
4. Shared Functional
5. Principal Angle
6. Overlapped Functional

后两条保留在同一文件中，是一个完整的负向结果链：Principal Angle 检查几何重合，Overlapped Functional 直接检查把候选更新投影到 context span 后能否保留功能。几何重合本身不是功能恢复的保证。

Heat map 和 STI 分别位于 `heatmap.py` 与 `sti.py`，不会进入 `spectra.py`。

## 2. 固定实验契约

- 任务顺序：`Rest, Motor, P300, SSS, TS`
- 每任务两个独立 replicate，共 10 次 full fine-tuning
- 每次从同一 LaBraM pretrained backbone `W0` 开始
- Session 1、4 s、200 Hz；输入从 `[N,64,800]` reshape 为 `[N,64,4,200]`
- 输入已全局 z-score；不 `/100`，不做第二次标准化
- exact input positions：CLS `0` + EEG `1..64`
- AdamW；backbone LR `1e-4`，head LR `1e-3`，weight decay `0`
- class-weighted cross entropy，batch size `32`
- AMP、gradient clipping、scheduler、warmup、early stopping 均关闭
- 所有 final run 固定 20 epochs
- 每个 run 保存完整 adapted checkpoint 和 12 blocks × 6 matrices = 72 个 `ΔW`

所有 bACC functional spectrum 都在同一 pooled fitted dataset 上测量，表示 fitted task function 的保留程度，**不是 held-out generalisation**。

## 3. 环境

推荐直接使用服务器上已经能运行原始 LaBraM 的 PyTorch/CUDA/timm 环境，避免无意升级模型依赖。随后安装本包与轻量依赖：

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e . --no-deps
```

代码在 import 时不会强制载入 PyTorch；`selftest` 和纯几何单元测试只需要 NumPy。

## 4. 配置

先复制并编辑：

```bash
cp configs/reported_results.json configs/local.json
```

至少确认四个路径：

- `root`：十个训练 run 的新结果根目录
- `data_root`：五个 M3CV H5 文件所在目录
- `checkpoint`：`labram-base.pth`
- `modeling_file`：原始 LaBraM `modeling_finetune.py`

`reported_results.json` 是较实际的 aligned-replicate 分析：预设稀疏 coarse grid、2% crossing refinement、128 次随机基线、随机上界 q95。`full_analysis.json` 使用所有 replicate pairings、5% coarse grid、1% refinement、1000 次随机基线、随机上界 q99，计算量显著更大。

## 5. 推荐执行顺序

```bash
# 纯几何自测，不读数据、不载入模型
bash run.sh selftest

# 完整数据、checkpoint、模型与 channel mapping 预检
bash run.sh preflight --config configs/local.json

# 5 tasks × 2 replicates，全部从同一 W0 训练
bash run.sh train --config configs/local.json

# 检查 exact 10-run grid、哈希与 72-matrix archive
bash run.sh audit --config configs/local.json

# 六条 spectra
bash run.sh spectra --group all --config configs/local.json

# 两项独立分析
bash run.sh heatmap --config configs/local.json
bash run.sh sti --config configs/local.json

# 只从现有 CSV 重画
bash run.sh plot --config configs/local.json
```

从空目录一次运行全部阶段：

```bash
bash run.sh all --config configs/local.json
```

默认 `--resume` 只接受 scientific signature 一致的中间状态。改变 profile、随机次数、输入 delta 或 cutoff 后，旧 resume marker 会被拒绝。若要重训已存在目录，必须显式使用 `--no-resume --overwrite`；先人工确认目标目录。

## 6. 输出布局

```text
<root>/
  runs/<task>/rep01|rep02/
    adapted_checkpoint.pth
    delta_weights.npz
    epoch_metrics.csv
    metadata.json
    _SUCCESS.json
  run_manifest.json
  analysis/
    spectra/
    heatmap/
    sti/
    figures/
```

数值定义见 [METHODS.md](METHODS.md)，逐表字段和端点检查见 [OUTPUTS.md](OUTPUTS.md)。

## 7. 验证代码本身

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
python3 -m compileall -q labram_tsv tests
```

单元测试覆盖显式 vectorisation 对照、principal projection 恒等式、六谱端点、STI 正交/重合边界，以及 `spectra.py` 不导入 heat map、STI 或绘图的结构约束。
