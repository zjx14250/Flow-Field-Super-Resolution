import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from basicsr.utils.constants import CASE1_MEAN, CASE1_STD
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Case1Dataset(Dataset):
    """Dataset for case1 HDF5 data.

    Args:
        opt (dict): Config for dataset. It contains:
            dataroot_lq (str): Path to LQ (low quality) h5 file (case1_cu_grid.h5)
            dataroot_gt (str): Path to GT (ground truth) h5 file (case1_chao_grid.h5)
            phase (str): 'train' or 'val'
    """

    def __init__(self, opt):
        super(Case1Dataset, self).__init__()
        self.opt = opt
        self.phase = opt['phase']

        # Load data
        self.lq_path = opt['dataroot_lq']
        self.gt_path = opt['dataroot_gt']

        # Open HDF5 files
        self.lq_file = h5py.File(self.lq_path, 'r')
        self.gt_file = h5py.File(self.gt_path, 'r')

        # Get sample keys (sample_0, sample_1, ..., sample_199)
        self.sample_keys = [k for k in self.lq_file.keys() if k.startswith('sample_')]
        self.sample_keys.sort(key=lambda x: int(x.split('_')[1]))

        # Random split train/val (use 90% for training, 10% for validation)
        # Use fixed seed for reproducibility
        np.random.seed(42)
        indices = np.arange(len(self.sample_keys))
        np.random.shuffle(indices)

        n_samples = len(self.sample_keys)
        n_train = int(n_samples * 0.9)

        if self.phase == 'train':
            train_indices = indices[:n_train]
            self.sample_keys = [self.sample_keys[i] for i in train_indices]
        else:  # val
            val_indices = indices[n_train:]
            self.sample_keys = [self.sample_keys[i] for i in val_indices]

        # Global normalization statistics (computed from all resolutions)
        # Ensures consistent scale across cu/zhong/xi/chao datasets
        self.lq_mean = CASE1_MEAN
        self.lq_std = CASE1_STD
        self.gt_mean = CASE1_MEAN
        self.gt_std = CASE1_STD

    def __getitem__(self, index):
        # Get sample key
        sample_key = self.sample_keys[index]

        # Load LQ and GT data
        # Shape: (H, W, C)
        lq = self.lq_file[sample_key][:]  # (32, 64, 2)
        gt = self.gt_file[sample_key][:]  # (256, 512, 2)

        # Normalize
        lq = (lq - self.lq_mean) / self.lq_std
        gt = (gt - self.gt_mean) / self.gt_std

        # Convert to torch tensors and permute to (C, H, W)
        lq = torch.from_numpy(lq.astype(np.float32)).permute(2, 0, 1)  # (2, 32, 64)
        gt = torch.from_numpy(gt.astype(np.float32)).permute(2, 0, 1)  # (2, 256, 512)

        return {
            'lq': lq,
            'gt': gt,
            'lq_path': f'{self.lq_path}:{sample_key}',
            'gt_path': f'{self.gt_path}:{sample_key}'
        }

    def __len__(self):
        return len(self.sample_keys)

    def __del__(self):
        # Close HDF5 files
        if hasattr(self, 'lq_file'):
            self.lq_file.close()
        if hasattr(self, 'gt_file'):
            self.gt_file.close()
