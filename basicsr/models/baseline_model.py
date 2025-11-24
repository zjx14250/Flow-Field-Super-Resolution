"""Baseline model for parameter-free methods (interpolation baselines).

This model extends SRModel but skips training-related initialization,
only performing validation to compute metrics.
"""

import numpy as np
import torch
from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY
import os
import os.path as osp
from tqdm import tqdm
from basicsr.metrics import calculate_metric


# Import normalization constants from case2_sr_model
from basicsr.models.case2_sr_model import NORM_MEAN, NORM_STD, DATA_MIN, DATA_MAX


@MODEL_REGISTRY.register()
class BaselineModel(SRModel):
    """Model for parameter-free baseline methods (nearest, bilinear, bicubic interpolation).

    Skips training initialization and only performs validation.
    """

    def init_training_settings(self):
        """Skip training settings initialization for baseline methods."""
        # No optimizer, no losses needed for parameter-free methods
        # Initialize log_dict for compatibility with BasicSR training loop
        self.log_dict = {}

    def optimize_parameters(self, current_iter):
        """No optimization needed for parameter-free methods."""
        pass

    def get_current_learning_rate(self):
        """Return empty learning rate list for parameter-free methods."""
        return []

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """No learning rate update needed for parameter-free methods."""
        pass

    def save(self, epoch, current_iter):
        """No need to save models for parameter-free methods."""
        pass

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Validation function for 2-channel data.

        Reuses the validation logic from Case2SRModel.
        """
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
        self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = visuals['result']

            # Denormalize: reverse the normalization
            sr_denorm = sr_img * NORM_STD + NORM_MEAN  # Back to original data range

            # Scale to [0, 255] for PSNR calculation
            data_range = DATA_MAX - DATA_MIN
            sr_scaled = ((sr_denorm - DATA_MIN) / data_range * 255.0).clamp(0, 255)

            # Convert to numpy: remove batch dim if exists, then CHW -> HWC
            sr_scaled = sr_scaled.squeeze()  # Remove singleton dimensions
            metric_data['img'] = sr_scaled.permute(1, 2, 0).detach().cpu().numpy()

            if 'gt' in visuals:
                gt_img = visuals['gt']
                # Same denormalization and scaling for ground truth
                gt_denorm = gt_img * NORM_STD + NORM_MEAN
                gt_scaled = ((gt_denorm - DATA_MIN) / data_range * 255.0).clamp(0, 255)
                gt_scaled = gt_scaled.squeeze()  # Remove singleton dimensions
                metric_data['img2'] = gt_scaled.permute(1, 2, 0).detach().cpu().numpy()
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                              f'{img_name}_{current_iter}.npy')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                  f'{img_name}_{self.opt["val"]["suffix"]}.npy')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                  f'{img_name}_{self.opt["name"]}.npy')

                # Save as numpy array for 2-channel data
                os.makedirs(osp.dirname(save_img_path), exist_ok=True)
                sr_img_np = sr_img.detach().cpu().float().numpy()
                np.save(save_img_path, sr_img_np)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
