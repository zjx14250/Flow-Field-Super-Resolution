"""Custom SR Model for 2-channel case2 data."""
import numpy as np
import torch
from basicsr.models.sr_model import SRModel
from basicsr.utils import imwrite, tensor2img
from basicsr.utils.constants import CASE2_MAX, CASE2_MEAN, CASE2_MIN, CASE2_STD
from basicsr.utils.registry import MODEL_REGISTRY


# Global normalization and range constants (computed from all 4 resolutions)
# Ensures physical correctness for energy spectrum metric
NORM_MEAN = CASE2_MEAN  # Average mean across cu/zhong/xi/chao
NORM_STD = CASE2_STD    # Average std across cu/zhong/xi/chao
DATA_MIN = CASE2_MIN    # Global min from all datasets
DATA_MAX = CASE2_MAX    # Global max from all datasets


@MODEL_REGISTRY.register()
class Case2SRModel(SRModel):
    """SR model for 2-channel scientific data (case2).

    Extends SRModel to handle 2-channel data without RGB conversion.
    """

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Validation function for 2-channel data.

        Overrides parent to skip RGB conversion for 2-channel tensors.
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

            # Denormalize: reverse the normalization (mean=0, std=0.3)
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


# Add missing imports
import os
import os.path as osp
from tqdm import tqdm
from basicsr.metrics import calculate_metric
