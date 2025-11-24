# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BasicSR** is a PyTorch-based open-source framework for image and video restoration tasks (super-resolution, denoising, deblurring, etc.). This repository operates in **local development mode** with BasicSR installed as an editable package (`pip install -e .`), allowing direct modification of framework internals.

**Current Branch**: `flow_field_sr_project` - Contains custom extensions for scientific flow field super-resolution built on top of BasicSR v1.4.2.

## Core Architecture

### Registry-Based Plugin System

BasicSR uses automatic component discovery through Python decorators and file naming conventions:

| Component | Location | File Pattern | Decorator | Registry |
|-----------|----------|--------------|-----------|----------|
| Dataset | `basicsr/data/` | `*_dataset.py` | `@DATASET_REGISTRY.register()` | DATASET_REGISTRY |
| Architecture | `basicsr/archs/` | `*_arch.py` | `@ARCH_REGISTRY.register()` | ARCH_REGISTRY |
| Model | `basicsr/models/` | `*_model.py` | `@MODEL_REGISTRY.register()` | MODEL_REGISTRY |
| Loss | `basicsr/losses/` | `*_loss.py` | `@LOSS_REGISTRY.register()` | LOSS_REGISTRY |
| Metric | `basicsr/metrics/` | `*_metric.py` | `@METRIC_REGISTRY.register()` | METRIC_REGISTRY |

**Discovery Mechanism**:
1. Each `__init__.py` in component directories uses `scandir()` to find files matching `*_<type>.py`
2. Files are dynamically imported at module initialization
3. Decorators register components into global registries
4. YAML configs reference components via `type: <RegisteredName>` fields

**Critical**: File naming conventions are mandatory. A file named `my_network.py` instead of `my_network_arch.py` will **not** be discovered.

### Project Structure

```
BasicSR/
├── basicsr/                    # Main package (installed in editable mode)
│   ├── archs/                  # Network architectures
│   │   ├── swinir_arch.py      # SwinIR Transformer (custom: supports 2-channel input)
│   │   ├── interpolation_arch.py  # Baseline interpolation methods (custom)
│   │   └── [50+ built-in architectures]
│   │
│   ├── data/                   # Dataset implementations
│   │   ├── case1_dataset.py    # HDF5 flow field loader (custom, 90/10 split)
│   │   ├── case2_dataset.py    # HDF5 flow field loader (custom, different normalization)
│   │   └── [20+ built-in datasets]
│   │
│   ├── models/                 # Training/inference logic
│   │   ├── case1_sr_model.py   # Overrides validation for 2-channel data (custom)
│   │   ├── case2_sr_model.py   # Similar to case1 with different denormalization (custom)
│   │   ├── baseline_model.py   # Validation-only model for baselines (custom)
│   │   └── [10+ built-in models including SRModel, VideoBaseModel, GANModel]
│   │
│   ├── losses/                 # Loss functions
│   │   ├── diffusion_simple.py # DDPM loss for DiT (custom)
│   │   └── [Built-in: L1Loss, CharbonnierLoss, PerceptualLoss, GANLoss]
│   │
│   ├── metrics/                # Evaluation metrics
│   │   ├── flow_metric.py      # RMSE + Pearson correlation (custom)
│   │   ├── spectrum_metric.py  # Energy spectrum via 2D FFT (custom)
│   │   └── [Built-in: PSNR, SSIM, NIQE, LPIPS]
│   │
│   ├── utils/                  # Utilities (logging, registry, dist training, etc.)
│   ├── ops/                    # CUDA ops (deformable conv, fused act, upfirdn2d)
│   ├── train.py                # Training entry point
│   └── test.py                 # Inference entry point
│
├── options/                    # YAML configuration files
│   ├── train/
│   │   ├── SwinIR/
│   │   │   ├── train_swinir_case1_x2.yml
│   │   │   ├── train_swinir_case1_x4.yml
│   │   │   ├── train_swinir_case1_x8.yml
│   │   │   ├── train_swinir_case2_x2.yml
│   │   │   ├── train_swinir_case2_x4.yml
│   │   │   └── train_swinir_case2_x8.yml
│   │   ├── DiT/
│   │   │   └── train_dit_minimal_case1_x4.yml
│   │   └── Baseline/
│   │       ├── baseline_bicubic_case2_x8.yml
│   │       ├── baseline_bilinear_case2_x8.yml
│   │       └── baseline_nearest_case2_x8.yml
│   └── test/                   # Inference configs and other examples
│       └── [...]
│
├── datasets/                   # Training data (HDF5 flow fields, 462 MB total)
│   ├── case1_cu_grid.h5        # Low-res (32×64×2 channels)
│   ├── case1_chao_grid.h5      # High-res (256×512×2 channels)
│   └── [6 more case1/case2 resolution variants]
│
├── experiments/                # Training outputs (auto-created, auto-archived)
│   └── <experiment_name>/
│       ├── models/             # Checkpoints (.pth, .state)
│       ├── visualization/      # Validation outputs (.npy for 2-channel, .png for RGB)
│       ├── tb_logger/          # TensorBoard logs
│       └── train_*.log         # Detailed logs
│
├── scripts/                    # Utility scripts (dataset prep, plot, etc.)
├── tests/                      # Unit tests
├── inference/                  # Inference-only scripts
└── docs/                       # Documentation
```

### YAML Configuration System

Configs use YAML with specific structure for registry lookups:

```yaml
name: SwinIR_Case1_x2              # Experiment name (creates experiments/<name>/)
model_type: Case1SRModel           # MODEL_REGISTRY lookup
scale: 2                           # Upsampling factor
num_gpu: 1                         # Multi-GPU: auto uses DistributedDataParallel

datasets:
  train:
    type: Case1Dataset             # DATASET_REGISTRY lookup
    dataroot_lq: datasets/case1_cu_grid.h5
    dataroot_gt: datasets/case1_zhong_grid.h5
    phase: train                   # Dataset uses this to split data
    batch_size_per_gpu: 16
    num_worker_per_gpu: 4

  val:
    type: Case1Dataset
    phase: val

network_g:                         # Generator network
  type: SwinIR_Custom              # ARCH_REGISTRY lookup
  upscale: 2
  in_chans: 2                      # Crucial: RGB=3, flow field=2

train:
  total_iter: 10000                # Total training iterations
  optim_g:
    type: Adam
    lr: !!float 2e-4               # Use !!float for scientific notation

  pixel_opt:
    type: L1Loss                   # LOSS_REGISTRY lookup
    loss_weight: 1.0

  warmup_iter: 1000                # LR warmup: lr * 0.1 → lr over 1000 iters
  ema_decay: 0.999                 # Exponential moving average (0 to disable)

val:
  val_freq: 500                    # Validate every N iterations
  save_img: false                  # true: save PNG (RGB), false: save NPY (2-channel)

  metrics:
    psnr:
      type: calculate_psnr         # METRIC_REGISTRY lookup
      crop_border: 2               # Crop pixels from edges
    energy_spectrum:
      type: calculate_energy_spectrum  # Custom metric
      crop_border: 2

logger:
  print_freq: 100                  # Log to console every N iters
  save_checkpoint_freq: 1000       # Save checkpoint every N iters
  use_tb_logger: true              # Enable TensorBoard
```

**Config → Code Mapping**:
- `model_type: Case1SRModel` → Instantiates class from `basicsr/models/case1_sr_model.py`
- `datasets.train.type: Case1Dataset` → Instantiates class from `basicsr/data/case1_dataset.py`
- `network_g.type: SwinIR_Custom` → Instantiates class from `basicsr/archs/swinir_arch.py`
- All `type` fields are case-sensitive registry keys

## Common Development Commands

### Environment Setup

```bash
# Install BasicSR in editable mode (required for local development)
pip install -e .

# Install with CUDA extensions (optional, for deformable conv, etc.)
BASICSR_EXT=True pip install -e .

# Install pre-commit hooks (flake8, yapf, isort)
pre-commit install
```

### Training

```bash
# Basic training
python basicsr/train.py -opt options/train/SwinIR/train_swinir_case1_x2.yml

# Debug mode (print every iter, validate every 8 iters)
python basicsr/train.py -opt options/train/SwinIR/train_swinir_case1_x2.yml --debug

# Auto-resume from latest checkpoint
python basicsr/train.py -opt options/train/SwinIR/train_swinir_case2_x8.yml --auto_resume

# Force resume from specific checkpoint
# Edit config: path.resume_state: experiments/<name>/training_states/10000.state
python basicsr/train.py -opt options/train/SwinIR/train_swinir_case2_x8.yml

# Multi-GPU training (automatic DistributedDataParallel)
# Edit config: num_gpu: 4
python basicsr/train.py -opt options/train/SwinIR/train_swinir_case1_x8.yml
```

### Testing/Inference

```bash
# Run inference on test dataset
python basicsr/test.py -opt options/test_swinir_case1_x2.yml

# Test with specific checkpoint
# Edit config: path.pretrain_network_g: experiments/<name>/models/net_g_10000.pth
python basicsr/test.py -opt options/test_swinir_case1_x2.yml
```

### Code Quality

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Format code (line length 120, PEP8 style)
yapf -i -r basicsr/

# Check style violations
flake8 basicsr/ --config=setup.cfg
```

### Experiment Management

```bash
# View TensorBoard logs
tensorboard --logdir experiments/<experiment_name>/tb_logger

# Check training progress
tail -f experiments/<experiment_name>/train_*.log

# List all checkpoints
ls -lh experiments/<experiment_name>/models/
```

## Development Workflow

### Adding a Custom Dataset

1. **Create `basicsr/data/my_dataset.py`**:

```python
import torch
from torch.utils.data import Dataset
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class MyDataset(Dataset):
    """Custom dataset for <description>.

    Args:
        opt (dict): Config with keys:
            dataroot_lq (str): Path to low-quality data
            dataroot_gt (str): Path to ground truth data
            phase (str): 'train' or 'val'
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # Load file paths, metadata, etc.

    def __getitem__(self, index):
        # Load and preprocess data
        # Must return dict with at least: {'lq': tensor, 'gt': tensor}
        return {
            'lq': lq_tensor,        # Shape: (C, H, W), dtype: float32
            'gt': gt_tensor,        # Shape: (C, H*scale, W*scale)
            'lq_path': str_path,    # For logging
            'gt_path': str_path
        }

    def __len__(self):
        return num_samples
```

2. **Reference in config**:

```yaml
datasets:
  train:
    type: MyDataset  # Must match registered name (case-sensitive)
    dataroot_lq: path/to/lq
    dataroot_gt: path/to/gt
```

### Adding a Custom Architecture

1. **Create `basicsr/archs/my_arch.py`**:

```python
import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class MyArch(nn.Module):
    """Custom architecture for <task>.

    Args:
        upscale (int): Upsampling factor (2, 3, 4, 8)
        in_chans (int): Number of input channels (3 for RGB, 2 for flow field)
        num_feat (int): Number of feature channels
    """

    def __init__(self, upscale=4, in_chans=3, num_feat=64, **kwargs):
        super().__init__()
        # Define layers
        self.conv_first = nn.Conv2d(in_chans, num_feat, 3, 1, 1)
        # ... more layers
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input LR image, shape (B, in_chans, H, W)

        Returns:
            Tensor: Output SR image, shape (B, in_chans, H*upscale, W*upscale)
        """
        feat = self.conv_first(x)
        # ... forward pass
        return output
```

2. **Reference in config**:

```yaml
network_g:
  type: MyArch
  upscale: 4
  in_chans: 3
  num_feat: 128
```

### Adding a Custom Model (Training Logic)

1. **Create `basicsr/models/my_model.py`**:

```python
import torch
from collections import OrderedDict
from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class MyModel(SRModel):
    """Custom training model for <specific use case>.

    Inherits from SRModel, which provides:
    - self.net_g: Generator network
    - self.lq, self.gt: Input data tensors
    - self.optimizer_g: Optimizer
    - setup_optimizers(), save(), validation(), etc.
    """

    def init_training_settings(self):
        """Initialize losses, optimizers (called once during setup)."""
        super().init_training_settings()

        # Add custom loss (example: multi-loss)
        train_opt = self.opt['train']
        if train_opt.get('custom_loss_opt'):
            from basicsr.models.losses import build_loss
            self.custom_loss = build_loss(train_opt['custom_loss_opt']).to(self.device)

    def feed_data(self, data):
        """Move data to GPU."""
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        """Training step: forward + backward + optimize."""
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        # Calculate losses
        l_total = 0
        loss_dict = OrderedDict()

        # Pixel loss
        l_pix = self.cri_pix(self.output, self.gt)
        l_total += l_pix
        loss_dict['l_pix'] = l_pix

        # Custom loss (if defined)
        if hasattr(self, 'custom_loss'):
            l_custom = self.custom_loss(self.output, self.gt)
            l_total += l_custom
            loss_dict['l_custom'] = l_custom

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    # Override validation for non-RGB data (example)
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Custom validation (e.g., skip RGB conversion for 2-channel data)."""
        # See basicsr/models/case1_sr_model.py for full example
        pass
```

2. **Common override scenarios**:
   - **Multi-loss training**: Add loss definitions in `init_training_settings()`
   - **Non-RGB data**: Override `nondist_validation()` to skip `tensor2img()` RGB conversion
   - **Custom validation**: Override `nondist_validation()` for specialized metric computation
   - **Validation-only model**: Skip training (see `basicsr/models/baseline_model.py`)

### Adding a Custom Metric

1. **Create `basicsr/metrics/my_metric.py`**:

```python
import numpy as np
from basicsr.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def calculate_my_metric(img, img2, crop_border=0, input_order='HWC', **kwargs):
    """Calculate custom metric.

    Args:
        img (ndarray): Predicted image, shape (H, W, C), range [0, 255]
        img2 (ndarray): Ground truth image, shape (H, W, C), range [0, 255]
        crop_border (int): Crop border pixels from edges
        input_order (str): 'HWC' or 'CHW'

    Returns:
        float: Metric value (higher is better for quality, lower for error)
    """
    assert img.shape == img2.shape, f'Shape mismatch: {img.shape} vs {img2.shape}'

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # Implement metric calculation
    metric_value = np.mean((img - img2) ** 2)  # Example: MSE
    return metric_value
```

2. **Reference in config**:

```yaml
val:
  metrics:
    my_metric:
      type: calculate_my_metric
      crop_border: 4
```

## Important Implementation Details

### Normalization Consistency (Critical for Custom Data)

For datasets using custom normalization (not [0, 1] or [0, 255]), **all locations must use identical parameters**:

1. **Dataset** (`basicsr/data/case1_dataset.py`): Normalize input
```python
self.lq_mean = 0.244449
self.lq_std = 0.266751
lq = (lq - self.lq_mean) / self.lq_std
```

2. **Model** (`basicsr/models/case1_sr_model.py`): Denormalize for validation
```python
NORM_MEAN = 0.244449  # MUST match dataset
NORM_STD = 0.266751
sr_denorm = sr_img * NORM_STD + NORM_MEAN
```

3. **Metrics** (`basicsr/metrics/spectrum_metric.py`): Use physical units
```python
NORM_MEAN = 0.244449  # MUST match dataset and model
NORM_STD = 0.266751
```

**Best Practice**: Create a `basicsr/utils/constants.py` file to centralize normalization parameters and avoid desynchronization.

### Experiment Output Management

**Auto-Archiving**: BasicSR automatically renames existing experiments when re-running:

```
experiments/
├── SwinIR_Case1_x8/                          # Latest run
├── SwinIR_Case1_x8_archived_20251115_120000/ # Previous run
└── SwinIR_Case1_x8_archived_20251115_150000/ # Earlier run
```

**Output Structure**:
```
experiments/<name>/
├── models/
│   ├── net_g_10000.pth      # Network weights only (for inference)
│   ├── 10000.state          # Full state: optimizer, scheduler, iter (for resume)
│   └── net_g_latest.pth     # Latest weights (symlink)
├── training_states/
│   └── 10000.state          # Full training states
├── visualization/
│   └── <val_dataset_name>/
│       ├── sample_0000.png  # RGB validation images
│       └── sample_0000.npy  # Non-RGB data (2-channel flow field)
├── tb_logger/               # TensorBoard logs
├── <config_name>.yml        # Config snapshot (reproducibility)
└── train_<name>_<timestamp>.log
```

### Training Data Flow

```
1. Config Loading:
   parse_options() → Reads YAML → Extracts 'type' fields

2. Component Registration (on import):
   basicsr/data/__init__.py → scandir('*_dataset.py') → import modules → @DATASET_REGISTRY.register()
   basicsr/archs/__init__.py → scandir('*_arch.py') → import modules → @ARCH_REGISTRY.register()
   basicsr/models/__init__.py → scandir('*_model.py') → import modules → @MODEL_REGISTRY.register()

3. Component Instantiation:
   build_dataset(dataset_opt) → DATASET_REGISTRY.get(dataset_opt['type'])(**dataset_opt)
   build_network(network_opt) → ARCH_REGISTRY.get(network_opt['type'])(**network_opt)
   build_model(opt) → MODEL_REGISTRY.get(opt['model_type'])(opt)

4. Training Loop:
   DataLoader → Dataset.__getitem__() → {'lq': tensor, 'gt': tensor}
   → Model.feed_data() → self.lq, self.gt to GPU
   → Model.optimize_parameters() → Forward + Loss + Backward
   → Every N iters: Model.validation() → Compute metrics → Save visualizations
   → Every M iters: Model.save() → Save checkpoints
```

## Key Built-in Utilities

### Data Utilities (`basicsr/data/`)
- **Degradations**: `degradations.py` (blur kernels, noise, JPEG compression, downsampling)
- **Transforms**: `transforms.py` (paired_random_crop, augment, random_augmentation)
- **I/O**: `data_util.py` (read_img_seq, img2tensor, tensor2img)

### Architecture Utilities (`basicsr/archs/`)
- **Blocks**: `arch_util.py` (ResidualBlockNoBN, Upsample, make_layer, PixelShufflePack)
- **Built-in Archs**: EDSR, RCAN, SRResNet, ESRGAN (RRDBNet), SwinIR, EDVR, BasicVSR, etc.

### Losses (`basicsr/models/losses/`)
- **Pixel**: L1Loss, MSELoss, CharbonnierLoss
- **Perceptual**: PerceptualLoss (VGG features), GANLoss
- **Custom**: Build with `build_loss(loss_opt)`

### Metrics (`basicsr/metrics/`)
- **Built-in**: calculate_psnr, calculate_ssim, calculate_niqe, calculate_lpips
- **Custom (this project)**: calculate_rmse, calculate_pearsonr, calculate_energy_spectrum

### Training Utilities (`basicsr/utils/`)
- **Registry**: `registry.py` (DATASET_REGISTRY, ARCH_REGISTRY, MODEL_REGISTRY, LOSS_REGISTRY, METRIC_REGISTRY)
- **Distributed**: `dist_util.py` (init_dist, get_dist_info)
- **Logger**: `logger.py` (get_root_logger, MessageLogger, AvgTimer)
- **Options**: `options.py` (parse_options, dict2str, copy_opt_file)

## Advanced Features

### Multi-GPU Training (Automatic DDP)

Set in config (no code changes needed):
```yaml
num_gpu: 4  # Automatically uses DistributedDataParallel
```

Launch with:
```bash
# Single-node multi-GPU (automatic)
python basicsr/train.py -opt options/train/SwinIR/train_swinir_case1_x8.yml

# Multi-node (manual distributed launch)
# Node 0: CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=<IP> --master_port=<PORT> basicsr/train.py -opt <config>
# Node 1: CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=<IP> --master_port=<PORT> basicsr/train.py -opt <config>
```

### EMA (Exponential Moving Average)

Stabilize inference by maintaining smoothed parameters:
```yaml
train:
  ema_decay: 0.999  # 0 to disable
```

### Learning Rate Scheduling

```yaml
train:
  scheduler:
    type: MultiStepLR
    milestones: [50000, 100000, 150000]
    gamma: 0.5

  warmup_iter: 1000  # Linearly increase LR: lr * 0.1 → lr over 1000 iters
```

### TensorBoard and WandB Logging

```yaml
logger:
  use_tb_logger: true

  wandb:
    project: my_project
    resume_id: <run_id>  # For resuming wandb logging
```

### Mixed Precision Training

Enable AMP (Automatic Mixed Precision) for faster training:
```yaml
train:
  use_amp: true  # Uses torch.cuda.amp
```

## Debugging and Troubleshooting

### Common Errors

1. **"Cannot find [ComponentType]"**
   - Cause: File not named with correct suffix (`_dataset.py`, `_arch.py`, etc.)
   - Fix: Rename file to match pattern and ensure `@REGISTRY.register()` decorator is present

2. **Shape mismatch during validation**
   - Cause: RGB conversion applied to non-RGB data (e.g., 2-channel flow field)
   - Fix: Override `nondist_validation()` in your model to skip `tensor2img()`

3. **NaN loss or wrong metrics**
   - Cause: Inconsistent normalization parameters across dataset/model/metrics
   - Fix: Verify `mean` and `std` values match exactly in all locations

4. **Out of memory**
   - Cause: Batch size too large or accumulating gradients
   - Fix: Reduce `batch_size_per_gpu` or use gradient accumulation

5. **"Resume training failed"**
   - Cause: Config mismatch between checkpoint and current config
   - Fix: Ensure network architecture and optimizer settings match checkpoint

### Debug Mode

Always test new components with:
```bash
python basicsr/train.py -opt options/my_config.yml --debug
```

This enables:
- Print every iteration (ignore `print_freq`)
- Validate every 8 iterations (ignore `val_freq`)
- Fast error detection

### Inspection Tips

1. **Check data shapes**: Add `print()` in `Dataset.__getitem__()`
2. **Verify registry**: Check `DATASET_REGISTRY._obj_map.keys()` in Python console
3. **Profile training**: Use `torch.profiler` or `line_profiler`
4. **Monitor GPU**: `watch -n 1 nvidia-smi`
5. **Check logs**: `experiments/<name>/train_*.log` contains full error traces

## Flow Field SR Project Specifics

### Custom Components for Scientific Data

This branch includes extensions for 2-channel flow field super-resolution:

- **Datasets**: `case1_dataset.py`, `case2_dataset.py` (HDF5 loaders with custom normalization)
- **Models**: `case1_sr_model.py`, `case2_sr_model.py` (override validation to skip RGB conversion)
- **Metrics**: `flow_metric.py` (RMSE, Pearson), `spectrum_metric.py` (energy spectrum via FFT)
- **Baselines**: `interpolation_arch.py` (Nearest/Bilinear/Bicubic), `baseline_model.py` (validation-only)

### Data Format

- **Input**: HDF5 files with shape `(N, H, W, 2)` where N = number of samples, channels = (u, v) velocity components
- **Resolution variants**: cu (32×64), zhong (64×128), xi (128×256), chao (256×512)
- **Normalization**: Case1 and Case2 use different mean/std computed from training data

### Training Example

```bash
# Train SwinIR for 8x SR on Case2 data
python basicsr/train.py -opt options/train/SwinIR/train_swinir_case2_x8.yml --auto_resume

# Evaluate baseline (no training, just validation)
python basicsr/train.py -opt options/train/Baseline/baseline_bicubic_case2_x8.yml
```

## Code Style Guidelines

BasicSR follows PEP8 with modifications (enforced by pre-commit hooks):

- **Line length**: 120 characters
- **Formatter**: yapf (based_on_style = pep8)
- **Import order**: isort (stdlib → third-party → local)
- **Linter**: flake8 (ignores W503, W504)

**Before committing**:
```bash
pre-commit run --all-files
```

## Notes

- **Editable install**: Changes to `basicsr/` take effect immediately (no reinstall needed)
- **Experiment archiving**: Old experiments are never overwritten, only renamed with timestamps
- **Multi-GPU**: Uses PyTorch DDP automatically when `num_gpu > 1`
- **CUDA ops**: Optional CUDA extensions (deformable conv, fused act, upfirdn2d) require `BASICSR_EXT=True` during install
- **Data format**: Tensors should be float32, range [0, 1] or [-1, 1] (avoid uint8 during training)
