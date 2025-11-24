"""DiT training model adapted for BasicSR framework.

Implements diffusion-based super-resolution training with simplified DDPM.
"""

import torch
import torch.nn.functional as F
from basicsr.models.base_model import BaseModel
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.archs import build_network
from basicsr.metrics import calculate_metric
import os.path as osp


@MODEL_REGISTRY.register()
class DiTSimpleModel(BaseModel):
    """Simplified DiT training model for quick validation.

    Trains a diffusion model to predict clean GT from noisy input,
    conditioned on low-resolution image.
    """

    def __init__(self, opt):
        super(DiTSimpleModel, self).__init__(opt)

        # Build network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # Load pretrained models if specified
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        """Initialize training settings: losses, optimizers, schedulers."""
        self.net_g.train()

        train_opt = self.opt['train']

        # Initialize diffusion process
        from losses.diffusion_simple import create_diffusion
        diff_opt = self.opt['diffusion']
        self.diffusion = create_diffusion(
            num_timesteps=diff_opt['num_timesteps'],
            beta_start=diff_opt.get('beta_start', 0.0001),
            beta_end=diff_opt.get('beta_end', 0.02),
            device=self.device
        )

        # MSE loss for denoising
        self.mse_loss = torch.nn.MSELoss()

        # Setup optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        """Setup optimizer from config."""
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        """Feed data to the model.

        Args:
            data: dict with 'lq' and 'gt' keys
        """
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        """Training step: forward + backward + optimize.

        Implements diffusion training:
        1. Sample random timestep t
        2. Add noise to GT: x_t = q(x_t | x_0)
        3. Predict x_0 from x_t using model
        4. Compute MSE loss
        """
        self.optimizer_g.zero_grad()

        # 1. Sample random timesteps for each sample in batch
        batch_size = self.gt.shape[0]
        t = torch.randint(
            0, self.diffusion.num_timesteps,
            (batch_size,), device=self.device, dtype=torch.long
        )

        # 2. Forward diffusion: add noise to GT
        noise = torch.randn_like(self.gt)
        gt_noisy = self.diffusion.q_sample(self.gt, t, noise)

        # 3. Predict clean GT from noisy version (conditioned on LQ)
        gt_pred = self.net_g(gt_noisy, t, self.lq)

        # 4. Compute loss (predict x_0, not noise)
        l_total = self.mse_loss(gt_pred, self.gt)

        # Backward and optimize
        l_total.backward()
        self.optimizer_g.step()

        # Log losses
        self.log_dict = {'l_mse': l_total.item()}

    def test(self):
        """Test (inference) mode: sample from diffusion model.

        Uses p_sample_loop to iteratively denoise from pure noise.
        """
        self.net_g.eval()
        with torch.no_grad():
            # Sample from diffusion (start from noise, denoise iteratively)
            self.output = self.diffusion.p_sample_loop(
                model=self.net_g,
                shape=self.gt.shape if hasattr(self, 'gt') else (self.lq.shape[0], self.lq.shape[1],
                                                                   self.lq.shape[2] * self.opt['scale'],
                                                                   self.lq.shape[3] * self.opt['scale']),
                lq=self.lq,
                progress=False,  # Disable progress bar during validation
                clip_denoised=True
            )
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Distributed validation (not implemented, uses nondist version)."""
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Non-distributed validation.

        Validates on the entire validation set and computes metrics.
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
            from tqdm import tqdm
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = visuals['result']

            # Denormalize (case1 uses mean=0.244449, std=0.266751)
            # For simplicity, assume normalized to [-1, 1], scale back to [0, 255]
            NORM_MEAN = 0.244449
            NORM_STD = 0.266751
            DATA_MIN = -0.492871
            DATA_MAX = 0.731227

            # sr_img is in [-1, 1] range from diffusion, denormalize
            sr_denorm = sr_img * NORM_STD + NORM_MEAN
            data_range = DATA_MAX - DATA_MIN
            sr_scaled = ((sr_denorm - DATA_MIN) / data_range * 255.0).clamp(0, 255)

            sr_scaled = sr_scaled.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            metric_data['img'] = sr_scaled

            if 'gt' in visuals:
                gt_img = visuals['gt']
                gt_denorm = gt_img * NORM_STD + NORM_MEAN
                gt_scaled = ((gt_denorm - DATA_MIN) / data_range * 255.0).clamp(0, 255)
                gt_scaled = gt_scaled.squeeze().permute(1, 2, 0).detach().cpu().numpy()
                metric_data['img2'] = gt_scaled
                del self.gt

            # Clean up memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                import numpy as np
                if self.opt['is_train']:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], img_name,
                        f'{img_name}_{current_iter}.npy'
                    )
                else:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.npy'
                    )

                import os
                os.makedirs(osp.dirname(save_img_path), exist_ok=True)
                # Save as numpy for 2-channel data
                sr_img_np = sr_img.detach().cpu().float().numpy()
                np.save(save_img_path, sr_img_np)

            if with_metrics:
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

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        """Log validation metrics to logger and TensorBoard."""
        from basicsr.utils import get_root_logger

        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)

        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        """Return current visuals for saving/visualization."""
        out_dict = {'lq': self.lq.detach().cpu()}
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        if hasattr(self, 'output'):
            out_dict['result'] = self.output.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        """Save networks and training state."""
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
