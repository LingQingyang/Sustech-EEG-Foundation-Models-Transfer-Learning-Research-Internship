# Experiment A：跨被试判别谱分析

对应脚本：

```bash
run_experiment_A_discriminant_suite_v3.py
```

## 1. 功能说明

该脚本用于分析 EEG foundation model embedding 中是否存在可跨被试迁移的任务状态判别轴。

默认在 CogBCI 数据集上进行三分类：

- `0`：Resting
- `1`：N-Back
- `2`：MATB

默认采用 4-fold subject-heldout 设置。每一折中：

1. 仅使用训练被试拟合均值；
2. 仅使用训练被试拟合 SVD；
3. 仅使用训练被试拟合多分类 LDA 判别轴；
4. 将未见过的测试被试投影到训练得到的判别空间；
5. 根据训练集类别中心完成分类。

测试被试不会参与判别轴、SVD 或类别中心的拟合。

---

## 2. 输入数据格式

输入为一个 HDF5 文件，至少需要包含：

```text
embedding
task_id
subject_id
```

其中：

- `embedding`：EEG foundation model 输出的 embedding；
- `task_id`：任务标签；
- `subject_id`：被试编号。

如实际键名不同，可通过命令行参数修改：

```bash
--embedding-key
--label-key
--subject-key
```

---

## 3. 环境依赖

建议使用 Python 3.9 及以上版本。

主要依赖：

```bash
pip install numpy scipy h5py matplotlib
```

---

## 4. 基本运行示例

```bash
python3 run_experiment_A_discriminant_suite_v3.py \
  --h5 /path/to/cogbci_embeddings.h5 \
  --embedding-key embedding \
  --label-key task_id \
  --subject-key subject_id \
  --include-labels 0,1,2 \
  --model-name BIOT \
  --cv subject_kfold \
  --n-subject-folds 4 \
  --svd-dims 100,200,300,500 \
  --plot-svd-dims 500 \
  --n-shuffle 100 \
  --skip-subject-control \
  --outdir /path/to/output
```

如果只想先确认流程能否运行，可以减少维度和 shuffle 次数：

```bash
python3 run_experiment_A_discriminant_suite_v3.py \
  --h5 /path/to/cogbci_embeddings.h5 \
  --model-name BIOT \
  --include-labels 0,1,2 \
  --svd-dims 100 \
  --plot-svd-dims 100 \
  --n-shuffle 5 \
  --skip-subject-control \
  --outdir /path/to/smoke_test
```

---

## 5. 常用参数

### 跨被试划分

```bash
--cv subject_kfold
--n-subject-folds 4
```

也可以使用 leave-one-subject-out：

```bash
--cv loso
```

### SVD 维度

```bash
--svd-dims 100,200,300,500
```

### 标签打乱基线

```bash
--n-shuffle 100
```

`0` 表示不运行标签打乱检验。

### 仅运行主任务分类

```bash
--skip-subject-control
```

不加该参数时，脚本还会额外运行“被试身份分类”对照实验。

### 保存更详细的图和点坐标

```bash
--plot-each-subject
--save-test-points
--save-train-points
```

---

## 6. 主要输出

结果会保存在：

```text
<outdir>/<model_name>/
```

主要目录包括：

```text
state_probe/
├── tables/
├── plots_train/
├── plots_fold_test/
├── plots_subject_test/
├── plots_summary/
└── points/
```

其中常用结果包括：

```text
tables/summary_by_svd_dim.csv
tables/fold_metrics.csv
tables/subject_metrics.csv
tables/subject_class_ld_stats.csv
plots_summary/summary_balanced_acc_vs_svd_dim.png
plots_summary/confusion_matrix_M*.png
```

### 核心指标

- `test_balanced_acc_mean`：跨被试 balanced accuracy；
- `heldout_macro_auc_mean`：跨被试 macro AUC；
- `lambda1_mean`、`lambda2_mean`：主要判别轴对应的广义特征值；
- `p_bacc_median`、`p_macro_auc_median`：相对于标签打乱基线的经验 p 值。

三分类随机水平为：

```text
balanced accuracy = 1 / 3
```

---

## 7. 方法边界

该脚本分析的是冻结 EEG foundation model embedding 中的线性可分结构。

它不会更新 backbone 参数，也不是 LoRA 微调脚本。其主要目的包括：

- 判断是否存在可跨被试迁移的任务状态判别轴；
- 比较不同 foundation model embedding 的状态可分性；
- 分析个体偏移是否破坏跨被试分类；
- 为后续个体化适配提供几何诊断依据。
