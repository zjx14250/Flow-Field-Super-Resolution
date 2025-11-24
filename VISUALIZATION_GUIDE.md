# 流场超分辨率可视化系统使用指南

本文档说明如何使用流场超分辨率对比可视化系统，批量生成所有验证集样本的对比图。

---

## 系统概述

该系统为Case1和Case2流场数据提供全面的可视化对比功能，支持6种配置（Case1/2 × x2/x4/x8），对比5种方法：

1. **LQ (Nearest)** - 低分辨率输入的最近邻上采样（保留块状特征，展示原始低分辨率）
2. **GT** - 真实高分辨率数据
3. **Bilinear** - 双线性插值基线
4. **Bicubic** - 双三次插值基线
5. **SwinIR** - Swin Transformer超分辨率模型

**为什么使用Nearest上采样显示LQ？**
- LQ原始尺寸（32×64）远小于GT（如256×512）
- 为了在同一grid中对比，需要将LQ上采样到GT尺寸
- 使用**Nearest插值**保留块状/模糊特征，直观展示低分辨率的"像素感"
- 如果用Bicubic上采样，LQ会看起来过于平滑，无法体现分辨率差异

每个样本生成6种可视化：
- **comparison_quiver.png** - 流场箭头图（5列对比）
- **comparison_heatmap_u.png** - U速度分量热图（5列对比）
- **comparison_heatmap_v.png** - V速度分量热图（5列对比）
- **error_heatmap.png** - 逐像素误差热图（3列：Bilinear/Bicubic/SwinIR vs GT）
- **error_histogram.png** - 误差统计直方图（3个方法，带RMSE/MAE标注）
- **energy_spectrum.png** - 2D FFT能量谱对比曲线（5条曲线）

---

## 快速开始

### 1. 环境检查

确保已安装BasicSR及依赖：
```bash
# 检查BasicSR是否已安装
python -c "import basicsr; print('BasicSR installed')"

# 检查必需的库
python -c "import matplotlib, numpy, torch, h5py; print('All dependencies OK')"
```

### 2. 数据和模型检查

确保数据集和训练好的checkpoint存在：
```bash
# 检查数据集
ls -lh datasets/case1_cu_grid.h5 datasets/case1_zhong_grid.h5

# 检查SwinIR checkpoint（以Case1 x2为例）
ls -lh experiments/SwinIR_Case1_x2/models/
```

### 3. 生成单个配置的可视化

```bash
# 生成Case1 x2的所有验证集样本可视化（约20个样本）
python scripts/visualize_flow_comparison.py \
    --case case1 \
    --scale 2 \
    --device cuda:0
```

生成的文件保存在：`experiments/visualizations/Case1_x2/`

---

## 详细使用方法

### 命令行参数

```bash
python scripts/visualize_flow_comparison.py \
    --case <case1|case2> \           # 必选：数据集类型
    --scale <2|4|8> \                # 必选：上采样倍数
    [--sample_ids <indices>] \       # 可选：样本索引（默认全部）
    [--output_dir <path>] \          # 可选：输出目录（默认experiments/visualizations）
    [--device <cuda:0|cpu>] \        # 可选：设备（默认cuda:0）
    [--skip_swinir]                  # 可选：跳过SwinIR推理
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--case` | 必选 | - | 数据集类型：`case1` 或 `case2` |
| `--scale` | 必选 | - | 上采样倍数：`2`, `4`, `8` |
| `--sample_ids` | 可选 | "" (全部) | 逗号分隔的样本索引，如 `0,1,2`（0-based，相对于验证集） |
| `--output_dir` | 可选 | `experiments/visualizations` | 输出根目录 |
| `--device` | 可选 | `cuda:0` | PyTorch设备，自动回退到CPU |
| `--skip_swinir` | 可选 | False | 跳过SwinIR推理，仅生成baseline对比 |

---

## 使用示例

### 示例1：生成Case1 x4的所有样本

```bash
python scripts/visualize_flow_comparison.py \
    --case case1 \
    --scale 4 \
    --device cuda:0
```

**说明**：
- 处理所有验证集样本（约20个）
- 使用GPU进行SwinIR推理
- 输出到 `experiments/visualizations/Case1_x4/`

---

### 示例2：仅可视化前5个样本

```bash
python scripts/visualize_flow_comparison.py \
    --case case2 \
    --scale 2 \
    --sample_ids 0,1,2,3,4 \
    --device cuda:0
```

**说明**：
- 仅处理验证集的前5个样本（索引0-4）
- 适用于快速测试或论文展示特定样本

---

### 示例3：仅生成baseline对比（无SwinIR）

```bash
python scripts/visualize_flow_comparison.py \
    --case case1 \
    --scale 8 \
    --skip_swinir \
    --device cpu
```

**说明**：
- 跳过SwinIR推理，仅对比Nearest/Bilinear/Bicubic
- 使用CPU（不需要GPU）
- 适用于：
  - SwinIR checkpoint不存在
  - 快速生成baseline对比
  - 无GPU环境

---

### 示例4：批量生成所有6种配置

```bash
# 方法1：使用批量脚本（推荐）
bash scripts/batch_visualize_all.sh

# 方法2：手动循环
for case in case1 case2; do
    for scale in 2 4 8; do
        python scripts/visualize_flow_comparison.py \
            --case $case \
            --scale $scale \
            --device cuda:0
    done
done
```

**说明**：
- 生成所有6种配置：Case1/Case2 × x2/x4/x8
- 每个配置约20个样本 × 6种可视化
- 总计约720张图片，1.5-2GB磁盘空间
- 预计耗时：15-30分钟（GPU）或1-2小时（CPU）

---

### 示例5：自定义输出目录

```bash
python scripts/visualize_flow_comparison.py \
    --case case1 \
    --scale 2 \
    --output_dir results/figures \
    --device cuda:0
```

**说明**：
- 输出到 `results/figures/Case1_x2/` 而非默认路径
- 适用于论文准备或多次实验对比

---

## 输出目录结构

```
experiments/visualizations/
├── Case1_x2/                          # Case1 x2配置
│   ├── sample_0/                      # 验证集第1个样本
│   │   ├── comparison_quiver.png      # 流场箭头图（6列，约400KB）
│   │   ├── comparison_heatmap_u.png   # U分量热图（6列，约170KB）
│   │   ├── comparison_heatmap_v.png   # V分量热图（6列，约160KB）
│   │   ├── error_heatmap.png          # 误差热图（5列，约130KB）
│   │   ├── error_histogram.png        # 误差统计（约130KB）
│   │   └── energy_spectrum.png        # 能量谱曲线（约230KB）
│   ├── sample_1/
│   ├── sample_2/
│   └── ... sample_19/
├── Case1_x4/
├── Case1_x8/
├── Case2_x2/
├── Case2_x4/
└── Case2_x8/
```

**单个样本的6张图总大小**：约1.0MB
**单个配置（20样本）总大小**：约20MB
**全部6个配置总大小**：约120MB

---

## 可视化图表说明

### 1. comparison_quiver.png（流场箭头图）
- **内容**：5列并排的流场箭头图（LQ | GT | Bilinear | Bicubic | SwinIR）
- **特点**：
  - 第一列LQ使用**Nearest上采样**，展示块状/像素化的低分辨率特征
  - 箭头方向表示流动方向
  - 箭头颜色表示速度幅值（viridis色图）
  - 高分辨率时自动稀疏采样（避免箭头重叠）
- **用途**：直观对比LQ的块状特征 vs GT的精细结构 vs 各方法的SR效果

### 2. comparison_heatmap_u.png / comparison_heatmap_v.png
- **内容**：U/V速度分量的热图对比
- **特点**：
  - 蓝-白-红发散色图（RdBu_r）
  - 自动对称colorbar范围（以0为中心）
  - 物理单位（已反归一化）
- **用途**：定量比较各方法的速度场数值

### 3. error_heatmap.png
- **内容**：3种SR方法（Bilinear/Bicubic/SwinIR）vs GT的误差
- **特点**：
  - 使用inferno色图（黑-红-黄）
  - 共享colorbar范围（可直接对比）
  - 误差 = √((u_pred - u_gt)² + (v_pred - v_gt)²)
  - **不包含LQ**：因为LQ是输入，不是SR方法
- **用途**：定位各SR方法的误差分布区域

### 4. error_histogram.png
- **内容**：3种SR方法的误差统计直方图
- **特点**：
  - 每个方法一个子图
  - 标注RMSE和MAE数值
  - 共享X轴范围（便于对比）
- **用途**：量化各SR方法的误差分布和统计特性

### 5. energy_spectrum.png
- **内容**：5种方法的2D FFT能量谱对比（LQ、GT、Bilinear、Bicubic、SwinIR）
- **特点**：
  - Log-log坐标
  - 径向平均能量 E(K) vs 波数 K
  - 所有方法在同一图上
  - **LQ曲线**展示低分辨率的能量分布（高频信息缺失）
- **用途**：评估各SR方法对湍流频谱结构的恢复能力

---

## 技术细节

### 归一化处理

系统使用集中管理的归一化参数（`basicsr/utils/constants.py`）：

| 数据集 | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Case1 | 0.244449 | 0.266751 | -0.492871 | 0.731227 |
| Case2 | 0.991759 | 1.087655 | -1.966761 | 2.925671 |

**处理流程**：
1. Dataset加载时归一化：`(x - mean) / std`
2. 模型推理使用归一化数据
3. 可视化前反归一化：`x * std + mean`（恢复物理单位）

### Checkpoint加载逻辑

脚本自动处理以下情况：
1. **优先**：`net_g_latest.pth`
2. **回退**：数值最大的checkpoint（如 `net_g_10000.pth`）
3. **目录命名**：自动处理 `case1` → `SwinIR_Case1_x{scale}`

如果checkpoint不存在且未指定`--skip_swinir`，脚本会报错并提示路径。

### 分辨率对应关系

| Scale | LQ分辨率 | GT分辨率 | GT文件 |
|-------|----------|----------|--------|
| x2 | 32×64 | 64×128 | `{case}_zhong_grid.h5` |
| x4 | 32×64 | 128×256 | `{case}_xi_grid.h5` |
| x8 | 32×64 | 256×512 | `{case}_chao_grid.h5` |

---

## 常见问题

### Q1: 提示"FileNotFoundError: ckpt path not found"

**原因**：SwinIR checkpoint不存在。

**解决方法**：
```bash
# 方法1：使用 --skip_swinir 跳过SwinIR
python scripts/visualize_flow_comparison.py \
    --case case1 --scale 2 \
    --skip_swinir

# 方法2：检查checkpoint路径是否正确
ls experiments/SwinIR_Case1_x2/models/

# 方法3：训练SwinIR模型
python basicsr/train.py -opt options/train/SwinIR/train_swinir_case1_x2.yml
```

---

### Q2: 提示"FileNotFoundError: lq path not found"

**原因**：数据集文件不存在。

**解决方法**：
```bash
# 检查数据集是否存在
ls -lh datasets/

# 确保以下文件存在：
# - case1_cu_grid.h5（LQ，32×64）
# - case1_zhong_grid.h5（GT for x2，64×128）
# - case1_xi_grid.h5（GT for x4，128×256）
# - case1_chao_grid.h5（GT for x8，256×512）
# - 同理Case2的4个文件
```

---

### Q3: GPU内存不足（CUDA out of memory）

**解决方法**：
```bash
# 方法1：使用CPU
python scripts/visualize_flow_comparison.py \
    --case case1 --scale 8 \
    --device cpu

# 方法2：减小batch处理（脚本默认batch_size=1，已是最小）
# 方法3：跳过SwinIR（如果仅需baseline对比）
python scripts/visualize_flow_comparison.py \
    --case case1 --scale 8 \
    --skip_swinir
```

---

### Q4: matplotlib报错"Passing a Normalize instance simultaneously with vmin/vmax"

**原因**：旧版matplotlib版本不兼容。

**解决方法**：
```bash
# 升级matplotlib
pip install --upgrade matplotlib>=3.3.0
```

---

### Q5: 如何只生成特定的可视化（如只要箭头图）？

**解决方法**：修改 `scripts/visualize_flow_comparison.py`，注释掉不需要的绘图部分（第195-240行）。

示例（仅保留箭头图）：
```python
# 保留
fig = create_comparison_grid(images, titles, ncols=6, figsize=(24, 4))
save_fig(fig, os.path.join(sample_dir, "comparison_quiver.png"))

# 注释掉其他可视化
# fig_u = create_comparison_grid(...)
# fig_v = create_comparison_grid(...)
# fig_err = plot_error_heatmap(...)
# fig_hist = plot_error_histogram(...)
# fig_spec = plot_energy_spectrum(...)
```

---

### Q6: 如何提取数值数据而非图片？

**方法1**：修改脚本保存numpy数组
```python
# 在 scripts/visualize_flow_comparison.py 中添加
np.save(os.path.join(sample_dir, 'lq_up.npy'), lq_up_np)
np.save(os.path.join(sample_dir, 'gt.npy'), gt_np)
np.save(os.path.join(sample_dir, 'swinir.npy'), swinir_np)
```

**方法2**：直接读取HDF5文件
```python
import h5py
with h5py.File('datasets/case1_chao_grid.h5', 'r') as f:
    gt = f['sample_0'][:]  # shape: (256, 512, 2)
```

---

### Q7: 如何调整图片分辨率/DPI？

修改 `basicsr/utils/flow_viz.py` 中的 `dpi` 参数（默认300）：

```python
# 降低DPI以减小文件大小
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)  # 原为300

# 或在保存时指定
fig.savefig(path, dpi=100, bbox_inches="tight")
```

---

## 性能优化

### 加速技巧

1. **使用GPU**（默认）：比CPU快10-20倍
   ```bash
   --device cuda:0
   ```

2. **减少样本数**：先测试少量样本
   ```bash
   --sample_ids 0,1,2
   ```

3. **跳过SwinIR**：仅生成baseline对比
   ```bash
   --skip_swinir
   ```

4. **并行处理多个配置**（需多GPU）：
   ```bash
   # 终端1
   CUDA_VISIBLE_DEVICES=0 python scripts/visualize_flow_comparison.py \
       --case case1 --scale 2 --device cuda:0 &

   # 终端2
   CUDA_VISIBLE_DEVICES=1 python scripts/visualize_flow_comparison.py \
       --case case2 --scale 2 --device cuda:0 &
   ```

### 预计耗时

| 配置 | 样本数 | GPU (V100) | CPU (16核) |
|------|--------|-----------|-----------|
| 单个配置 | 20 | 2-3分钟 | 10-15分钟 |
| 全部6个配置 | 120 | 15-20分钟 | 1-1.5小时 |

---

## 进阶用法

### 集成到Python脚本

```python
from basicsr.data.case1_dataset import Case1Dataset
from basicsr.utils.constants import denormalize, get_norm_params
from basicsr.utils.flow_viz import plot_flow_quiver, plot_energy_spectrum
import matplotlib.pyplot as plt

# 加载数据
dataset = Case1Dataset({
    'dataroot_lq': 'datasets/case1_cu_grid.h5',
    'dataroot_gt': 'datasets/case1_chao_grid.h5',
    'phase': 'val'
})
sample = dataset[0]

# 反归一化
mean, std, _, _ = get_norm_params('case1')
gt_np = sample['gt'].permute(1, 2, 0).numpy()  # (H, W, C)
gt_phys = denormalize(gt_np, mean, std)

# 绘制箭头图
fig = plot_flow_quiver(
    gt_phys[:, :, 0],  # U分量
    gt_phys[:, :, 1],  # V分量
    title='Ground Truth Flow Field'
)
plt.savefig('custom_visualization.png', dpi=300)
```

### 自定义绘图函数

参考 `basicsr/utils/flow_viz.py` 中的函数签名，可以轻松扩展：

```python
from basicsr.utils.flow_viz import plot_flow_heatmap

# 自定义色图和范围
fig = plot_flow_heatmap(
    data=u_component,
    title='Custom U Velocity',
    cmap='coolwarm',  # 使用不同色图
    vmin=-1.0,        # 固定范围
    vmax=1.0
)
```

---

## 文件清单

### 新增文件
| 文件 | 行数 | 说明 |
|------|------|------|
| `basicsr/utils/constants.py` | 69 | 归一化常量集中管理 |
| `basicsr/utils/flow_viz.py` | 313 | 流场可视化工具库 |
| `scripts/visualize_flow_comparison.py` | 254 | 主可视化脚本 |
| `scripts/batch_visualize_all.sh` | 38 | 批量执行脚本 |

### 修改文件
| 文件 | 修改内容 |
|------|----------|
| `basicsr/data/case1_dataset.py` | 导入并使用`CASE1_MEAN/STD` |
| `basicsr/data/case2_dataset.py` | 导入并使用`CASE2_MEAN/STD` |
| `basicsr/models/case1_sr_model.py` | 导入并使用`CASE1_*`常量 |
| `basicsr/models/case2_sr_model.py` | 导入并使用`CASE2_*`常量 |
| `basicsr/metrics/spectrum_metric.py` | 导入并使用`CASE1_MEAN/STD` |

---

## 引用和致谢

本可视化系统基于以下工具构建：
- **BasicSR** - 图像超分辨率框架
- **Matplotlib** - 科学可视化
- **PyTorch** - 深度学习框架
- **SwinIR** - Transformer超分辨率模型

如在论文中使用本可视化系统，请引用BasicSR和SwinIR：

```bibtex
@inproceedings{wang2021basicsr,
  title={BasicSR: Open Source Image and Video Restoration Toolbox},
  author={Wang, Xintao and others},
  booktitle={ACM MM},
  year={2021}
}

@inproceedings{liang2021swinir,
  title={SwinIR: Image Restoration Using Swin Transformer},
  author={Liang, Jingyun and others},
  booktitle={ICCV},
  year={2021}
}
```

---

## 支持和反馈

如遇到问题或需要功能扩展，请：
1. 检查本文档的"常见问题"章节
2. 查看代码注释（所有函数均有完整docstring）
3. 提交Issue到项目仓库

**生成日期**：2025-11-19
**文档版本**：1.0
