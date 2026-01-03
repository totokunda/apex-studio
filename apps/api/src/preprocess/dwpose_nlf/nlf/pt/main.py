import os

from nlf.pt import init

"separator"
import bisect
import itertools

import florch.callbacks
import numpy as np
import posepile.datasets2d as ds2d
import posepile.datasets3d as ds3d
import simplepyutils as spu
import torch
import torch.optim as optim
from simplepyutils import FLAGS, logger

import nlf.pt.backbones.builder as backbone_builder
import nlf.pt.models.field as lf_field
import nlf.pt.models.nlf_model as lf_model
import nlf.pt.models.nlf_trainer as lf_trainer
import nlf.pt.ptu as ptu
import nlf.pt.render_callback as render_callback
from nlf.paths import DATA_ROOT, PROJDIR
from nlf.pt.loading.densepose import load_dense
from nlf.pt.loading.keypoints2d import load_2d
from nlf.pt.loading.keypoints3d import load_kp
from nlf.pt.loading.parametric import load_parametric
from nlf.pt.util import TEST, TRAIN, VALID



def main():
    init.initialize()

    if FLAGS.train:
        job = LocalizerFieldJob()
        job.train()

    if FLAGS.render:
        job = LocalizerFieldJob()
        job.restore()
        job.render()


class LocalizerFieldJob(florch.TrainingJob):
    def __init__(self):
        super().__init__(
            wandb_project=FLAGS.wandb_project,
            wandb_config=FLAGS,
            logdir=FLAGS.logdir,
            init_path=FLAGS.init_path,
            load_path=FLAGS.load_path,
            training_steps=FLAGS.training_steps,
            grad_accum_steps=FLAGS.grad_accum_steps,
            loss_scale=FLAGS.loss_scale if FLAGS.dtype == 'float16' else 1.0,
            dynamic_loss_scale=FLAGS.dynamic_loss_scale,
            ema_momentum=FLAGS.ema_momentum,
            finetune_in_inference_mode=FLAGS.finetune_in_inference_mode,
            validate_period=FLAGS.validate_period,
            checkpoint_dir=FLAGS.checkpoint_dir,
            checkpoint_period=FLAGS.checkpoint_period,
            multi_gpu=FLAGS.multi_gpu,
            seed=FLAGS.seed,
            n_completed_steps=FLAGS.completed_steps,
            stop_step=FLAGS.stop_step,
            clip_grad_norm_median_factor=FLAGS.clip_grad_norm_median_factor,
            clip_grad_norm_quantile=FLAGS.clip_grad_norm_quantile,
            clip_grad_norm_histsize=FLAGS.clip_grad_norm_histsize,
            norm_loss_factor=FLAGS.norm_loss_factor,
            norm_loss_start_step=FLAGS.norm_loss_start_step,
            norm_loss_ramp_steps=FLAGS.norm_loss_ramp_steps,
            workers=FLAGS.workers,
        )

    def build_data(self):
        ds_parts_param = self.get_parts_param()
        ds_parts3d = self.get_parts3d()
        ds_parts2d = dict(mpii_down=4, coco_=4, jrdb_down=4, posetrack_down=4, aic_down=4, halpe=4)
        ds_parts_dense = dict(densepose_coco_=14, densepose_posetrack_=7)

        # PARAMETRIC
        bad_paths_param = set(
            spu.load_pickle(f'{DATA_ROOT}/posepile_33ds_new/bad_paths_param.pkl')
        )
        dataset3d_param = ds3d.Pose3DDatasetBarecat(FLAGS.dataset3d, FLAGS.image_barecat_path)
        examples3d_param = get_examples(dataset3d_param, TRAIN)
        examples3d_param = [ex for ex in examples3d_param if ex.path not in bad_paths_param]
        example_sections_param, roundrobin_sizes_param = get_sections_and_sizes(
            examples3d_param, ds_parts_param
        )
        if FLAGS.dataset3d_kp is None:
            FLAGS.dataset3d_kp = f'{DATA_ROOT}/posepile_28ds/annotations_28ds.barecat'

        dataset3d_kp = ds3d.Pose3DDatasetBarecat(FLAGS.dataset3d_kp, FLAGS.image_barecat_path)
        huge8_2_joint_info = spu.load_pickle(f'{PROJDIR}/huge8_2_joint_info2.pkl')
        stream_parametric = self.build_roundrobin_stream(
            example_sections=example_sections_param,
            load_fn=load_parametric,
            extra_args=(
                FLAGS.num_surface_points,
                FLAGS.num_internal_points,
                TRAIN,
            ),
            batch_size=FLAGS.batch_size_parametric,
            roundrobin_sizes=roundrobin_sizes_param,
        )

        # 3D KEYPOINTS
        bad_paths = spu.load_pickle(f'{DATA_ROOT}/posepile_28ds/bad_annos_28ds.pkl')
        bad_paths2 = spu.load_pickle(f'{DATA_ROOT}/posepile_33ds_new/bad_paths_kp3d.pkl')
        bad_paths = set(bad_paths) | set(bad_paths2)
        bad_impaths = [
            'dna_rendering_downscaled/1_0018_05/02/000090.jpg',
            'dna_rendering_downscaled/1_0018_05/02/000079.jpg',
        ]
        bad_impaths = set(bad_impaths)
        examples3d = [
            ex
            for ex in dataset3d_kp.examples[0]
            if ex.path not in bad_paths and ex.image_path not in bad_impaths
        ]
        example_sections3d, roundrobin_sizes3d = get_sections_and_sizes(examples3d, ds_parts3d)
        stream_kp = self.build_roundrobin_stream(
            example_sections=example_sections3d,
            load_fn=load_kp,
            extra_args=(dataset3d_kp.joint_info, TRAIN),
            batch_size=FLAGS.batch_size,
            roundrobin_sizes=roundrobin_sizes3d,
        )

        # 2D KEYPOINTS
        dataset2d = ds2d.Pose2DDatasetBarecat(FLAGS.dataset2d, FLAGS.image_barecat_path)
        examples2d = [*dataset2d.examples[TRAIN], *dataset2d.examples[VALID]]
        example_sections2d, roundrobin_sizes2d = get_sections_and_sizes(examples2d, ds_parts2d)
        stream_2d = self.build_roundrobin_stream(
            example_sections=example_sections2d,
            load_fn=load_2d,
            extra_args=(dataset2d.joint_info, huge8_2_joint_info, TRAIN),
            batch_size=FLAGS.batch_size_2d,
            roundrobin_sizes=roundrobin_sizes2d,
        )

        # DENSEPOSE
        dataset_dense = ds2d.Pose2DDatasetBarecat(FLAGS.dataset_dense, FLAGS.image_barecat_path)
        example_sections_dense, roundrobin_sizes_dense = get_sections_and_sizes(
            dataset_dense.examples[TRAIN], ds_parts_dense
        )
        stream_dense = self.build_roundrobin_stream(
            example_sections=example_sections_dense,
            load_fn=load_dense,
            extra_args=(dataset_dense.joint_info, huge8_2_joint_info, TRAIN),
            batch_size=FLAGS.batch_size_densepose,
            roundrobin_sizes=[14, 7],
        )

        # COMBINE
        data_train = self.merge_streams_to_torch_loader_train(
            streams=[stream_parametric, stream_kp, stream_dense, stream_2d],
            batch_sizes=[
                FLAGS.batch_size_parametric,
                FLAGS.batch_size,
                FLAGS.batch_size_densepose,
                FLAGS.batch_size_2d,
            ],
        )

        # VALIDATION
        if FLAGS.validate_period:
            examples3d_val = get_examples(dataset3d_param, VALID)
            validation_steps = len(examples3d_val) // FLAGS.batch_size_test
            examples3d_val = examples3d_val[: validation_steps * FLAGS.batch_size_test]
            stream_val = self.build_stream(
                examples=examples3d_val,
                load_fn=load_parametric,
                extra_args=(
                    FLAGS.num_surface_points,
                    FLAGS.num_internal_points,
                    VALID,
                ),
                shuffle_before_each_epoch=False,
            )
            data_val = self.stream_to_torch_loader_test(stream_val, FLAGS.batch_size_test)
        else:
            validation_steps = 0
            data_val = None

        return data_train, data_val, validation_steps

    def build_model(self):
        backbone, normalizer, out_channels = backbone_builder.build_backbone()
        weight_field = lf_field.build_field()
        model = lf_model.NLFModel(backbone, weight_field, normalizer, out_channels)
        if not self.get_load_path():
            if FLAGS.load_backbone_from:
                logger.info(f'Loading backbone from {FLAGS.load_backbone_from}')
                missing_keys, unexpected_keys = backbone.load_state_dict(
                    torch.load(FLAGS.load_backbone_from, weights_only=False, map_location='cpu'),
                    strict=False,
                )

                if len(missing_keys) > 0:
                    logger.warning(f'Missing keys in backbone model state_dict: {missing_keys}')
                if len(unexpected_keys) > 0:
                    logger.warning(
                        f'Unexpected keys in backbone model state_dict: {unexpected_keys}'
                    )

                if 'model_state_dict' in unexpected_keys:
                    model_state = torch.load(
                        FLAGS.load_backbone_from, weights_only=False, map_location='cpu'
                    )['model_state_dict']
                    backbone_state = {
                        k.removeprefix('backbone.'): v
                        for k, v in model_state.items()
                        if k.startswith('backbone.')
                    }
                    missing_keys, unexpected_keys = backbone.load_state_dict(
                        backbone_state, strict=False
                    )
                    if len(missing_keys) > 0:
                        logger.warning(
                            f'Missing keys in backbone model state_dict: {missing_keys}'
                        )
                    if len(unexpected_keys) > 0:
                        logger.warning(
                            f'Unexpected keys in backbone model state_dict: {unexpected_keys}'
                        )

            weight_field.gps_net.load_state_dict(
                torch.load(
                    f'{PROJDIR}/lbo_mlp_512fourier_2048gelu_{FLAGS.field_posenc_dim}.pt',
                    weights_only=True,
                    map_location='cpu',
                )
            )
        return model

    def build_trainer(self, model, **kwargs):
        return lf_trainer.NLFTrainer(model, **kwargs, random_seed=FLAGS.seed)

    def build_optimizer(self):
        if FLAGS.dual_finetune_lr:
            return self.build_optimizer_dual()

        weight_decay = FLAGS.weight_decay / np.sqrt(self.training_steps) / FLAGS.base_learning_rate

        if FLAGS.backbone.startswith('dinov2'):
            main_backbone_params_items = [
                (n, x) for n, x in self.model.backbone.named_parameters() if 'patch_embed' not in n
            ]
            patch_embed_params = [
                x for n, x in self.model.backbone.named_parameters() if 'patch_embed' in n
            ]
            main_params = itertools.chain(
                main_backbone_params_items, self.model.heatmap_head.layer.named_parameters()
            )
        else:
            main_params = itertools.chain(
                self.model.backbone.named_parameters(),
                self.model.heatmap_head.layer.named_parameters(),
            )

        if 'customwdecay' in FLAGS.custom:
            decay_params = []
            no_decay_params = []
            for name, p in main_params:
                if name.endswith(".bias") or "norm" in name or "gamma" in name:
                    no_decay_params.append(p)
                else:
                    decay_params.append(p)
        else:
            decay_params = [p for n, p in main_params]
            no_decay_params = []

        if FLAGS.optimizer == 'adamw':
            optimizer_class = optim.AdamW
            kwargs = {}
        #elif FLAGS.optimizer == 'stableadamw':
        #    import optimi
        #    optimizer_class = optimi.StableAdamW
        #    kwargs = {}  #'weight_decouple': True, 'kahan_sum': True, 'eps': FLAGS.adam_eps}
        else:
            raise ValueError(f'Unknown optimizer: {FLAGS.optimizer}')

        parameter_groups_dict = [
            {
                "params": tuple(decay_params),
                "weight_decay": weight_decay,
                'lr': 1.0,
                'initial_lr': 1.0,
                **kwargs,
            },
            {
                "params": itertools.chain(
                    self.model.heatmap_head.weight_field.parameters(),
                    [self.model.canonical_lefts, self.model.canonical_centers],
                    no_decay_params,
                ),
                "weight_decay": 0.0,
                'lr': 1.0,
                'initial_lr': 1.0,
                **kwargs,
            },
        ]

        lr_functions = [
            self.learning_rate_schedule,
            self.learning_rate_schedule,
        ]

        if FLAGS.backbone.startswith('dinov2'):
            parameter_groups_dict.append(
                {
                    "params": patch_embed_params,
                    "weight_decay": weight_decay,
                    'lr': FLAGS.patch_embed_lr_factor,
                    'initial_lr': FLAGS.patch_embed_lr_factor,
                    **kwargs,
                }
            )
            lr_functions.append(self.learning_rate_schedule)

        optimizer = optimizer_class(
            parameter_groups_dict,
            lr=1.0,
            fused=True if FLAGS.optimizer == 'adamw' and FLAGS.dtype == 'bfloat16' else None,
            betas=(FLAGS.adam_beta1, FLAGS.adam_beta2),
            eps=FLAGS.adam_eps,
        )
        return optimizer, lr_functions

    def build_optimizer_dual(self):
        weight_decay = FLAGS.weight_decay / (self.training_steps**0.5) / FLAGS.base_learning_rate

        if FLAGS.backbone.startswith('dinov2'):
            main_backbone_params_items = [
                (n, x) for n, x in self.model.backbone.named_parameters() if 'patch_embed' not in n
            ]
            patch_embed_params = [
                x for n, x in self.model.backbone.named_parameters() if 'patch_embed' in n
            ]
            main_params = main_backbone_params_items
        else:
            main_params = self.model.backbone.named_parameters()

        if 'customwdecay' in FLAGS.custom:
            decay_params = []
            no_decay_params = []
            for name, p in main_params:
                if name.endswith(".bias") or "norm" in name or "gamma" in name:
                    no_decay_params.append(p)
                else:
                    decay_params.append(p)
        else:
            decay_params = [p for n, p in main_params]
            no_decay_params = []

        if FLAGS.optimizer == 'adamw':
            optimizer_class = optim.AdamW
            kwargs = {}
        #elif FLAGS.optimizer == 'stableadamw':
            # import pytorch_optimizer
            # optimizer_class = pytorch_optimizer.StableAdamW
            # import optimi
            #optimizer_class = optimi.StableAdamW
            #kwargs = {}  #'weight_decouple': True, 'kahan_sum': True, 'eps': FLAGS.adam_eps}
        else:
            raise ValueError(f'Unknown optimizer: {FLAGS.optimizer}')

        parameter_groups_dict = [
            {
                "params": decay_params,
                "weight_decay": weight_decay,
                'lr': 1.0,
                'initial_lr': 1.0,
                **kwargs,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
                'lr': 1.0,
                'initial_lr': 1.0,
                **kwargs,
            },
            {
                "params": self.model.heatmap_head.layer.parameters(),
                "weight_decay": weight_decay,
                'lr': 1.0,
                'initial_lr': 1.0,
                **kwargs,
            },
            {
                "params": itertools.chain(
                    self.model.heatmap_head.weight_field.parameters(),
                    [self.model.canonical_lefts, self.model.canonical_centers],
                ),
                "weight_decay": 0.0,
                'lr': 1.0,
                'initial_lr': 1.0,
                **kwargs,
            },
        ]

        lr_functions = [
            self.learning_rate_schedule_finetune_low,
            self.learning_rate_schedule_finetune_low,
            self.learning_rate_schedule,
            self.learning_rate_schedule,
        ]
        if FLAGS.backbone.startswith('dinov2'):
            parameter_groups_dict.append(
                {
                    "params": patch_embed_params,
                    "weight_decay": weight_decay,
                    'lr': FLAGS.patch_embed_lr_factor,
                    'initial_lr': FLAGS.patch_embed_lr_factor,
                    **kwargs,
                }
            )
            lr_functions.append(self.learning_rate_schedule_finetune_low)

        optimizer = optimizer_class(
            parameter_groups_dict,
            lr=1.0,
            fused=True if FLAGS.optimizer == 'adamw' else None,
            betas=(FLAGS.adam_beta1, FLAGS.adam_beta2),
            eps=FLAGS.adam_eps,
        )
        return optimizer, lr_functions

    def learning_rate_schedule(self, step: int) -> float:
        n_warmup_steps = FLAGS.lr_warmup_steps
        n_phase1_steps = (1 - FLAGS.lr_cooldown_fraction) * self.training_steps - n_warmup_steps
        n_phase2_steps = self.training_steps - n_warmup_steps - n_phase1_steps
        b = FLAGS.base_learning_rate

        if step < n_warmup_steps:
            return b * FLAGS.field_lr_factor * step / n_warmup_steps
        elif step < n_warmup_steps + n_phase1_steps:
            if (
                FLAGS.rewarm_start_step is not None
                and FLAGS.rewarm_start_step < step < FLAGS.rewarm_start_step + 10000
            ):
                factor = (step - FLAGS.rewarm_start_step) / 10000
            else:
                factor = 1.0

            return factor * exp_decay(
                step,
                start_step=0,
                initial_value=b * FLAGS.field_lr_factor,
                decay_rate=1 / 3,
                decay_steps=n_warmup_steps + n_phase1_steps,
            )
        else:
            if FLAGS.cosine_cooldown:
                return cos_decay(
                    step,
                    start_step=n_warmup_steps + n_phase1_steps,
                    initial_value=1.0,
                    final_value=0.0,
                    decay_steps=n_phase2_steps,
                ) * exp_decay(
                    step,
                    start_step=0,
                    initial_value=b * FLAGS.field_lr_factor,
                    decay_rate=1 / 3,
                    decay_steps=n_warmup_steps + n_phase1_steps,
                )
            return exp_decay(
                step,
                start_step=n_warmup_steps + n_phase1_steps,
                initial_value=b / 30.0,
                decay_rate=0.3,
                decay_steps=n_phase2_steps,
            )

    def learning_rate_schedule_finetune_low(self, step: int) -> float:
        n_frozen_steps = FLAGS.frozen_backbone_steps
        n_warmup_steps = FLAGS.backbone_warmup_steps

        n_phase1_steps = (
            (1 - FLAGS.lr_cooldown_fraction) * self.training_steps
            - n_frozen_steps
            - n_warmup_steps
        )
        n_phase2_steps = self.training_steps - n_frozen_steps - n_warmup_steps - n_phase1_steps
        b = FLAGS.base_learning_rate

        if step < n_frozen_steps:
            return 0.0
        elif step < n_frozen_steps + n_warmup_steps:
            return b * FLAGS.backbone_lr_factor * (step - n_frozen_steps) / n_warmup_steps
        elif step < n_frozen_steps + n_warmup_steps + n_phase1_steps:
            return exp_decay(
                step,
                start_step=0,
                initial_value=b * FLAGS.backbone_lr_factor,
                decay_rate=1 / 3,
                decay_steps=n_frozen_steps + n_warmup_steps + n_phase1_steps,
            )
        else:
            if FLAGS.cosine_cooldown:
                return cos_decay(
                    step,
                    start_step=n_warmup_steps + n_frozen_steps + n_phase1_steps,
                    initial_value=1.0,
                    final_value=0.0,
                    decay_steps=n_phase2_steps,
                ) * exp_decay(
                    step,
                    start_step=0,
                    initial_value=b * FLAGS.backbone_lr_factor,
                    decay_rate=1 / 3,
                    decay_steps=n_frozen_steps + n_warmup_steps + n_phase1_steps,
                )
            else:
                return exp_decay(
                    step,
                    start_step=n_warmup_steps + n_frozen_steps + n_phase1_steps,
                    initial_value=b / 30.0,
                    decay_rate=0.3,
                    decay_steps=n_phase2_steps,
                )

    def build_callbacks(self):
        cbacks = []

        if FLAGS.render_callback_period != 0:
            cbacks.append(
                render_callback.RenderPredictionCallback(
                    start_step=500, interval=FLAGS.render_callback_period
                )
            )
        if FLAGS.batch_renorm:
            cbacks.append(AdjustRenormClipping(7500, 10000))

        if FLAGS.constrain_kernel_norm:
            cbacks.append(
                florch.callbacks.ConvMinMaxNormConstraint(
                    rate=FLAGS.constraint_rate, max_value=FLAGS.constrain_kernel_norm
                )
            )

        cbacks.append(
            florch.callbacks.MinMaxNormConstraint(
                [self.model.canonical_lefts, self.model.canonical_centers], max_value=0.07
            )
        )

        if FLAGS.unfreeze_parts_step:
            cbacks.append(
                florch.callbacks.FreezeLayers(
                    [
                        self.model.heatmap_head.weight_field.gps_net,
                        self.model.canonical_lefts,
                        self.model.canonical_centers,
                    ],
                    FLAGS.unfreeze_parts_step,
                )
            )
        #
        # cbacks.append(
        #     florch.callbacks.FreezeLayers([self.model.backbone], FLAGS.frozen_backbone_steps)
        # )

        if FLAGS.transition_bn:
            cbacks.append(florch.callbacks.TransitionBatchNorm(5000, 10000))

        return cbacks

    def get_parts3d(self):
        if 'subset_real' in FLAGS.custom:
            ds_parts3d = {
                'h36m_': 10,
                'muco_downscaled': 6,
                'humbi': 5,
                '3doh_down': 3,
                'panoptic_': 7,
                'aist_': 6,
                'aspset_': 4,
                'gpa_': 4,
                'bml_movi': 5,
                'mads_down': 2,
                'umpm_down': 2,
                'bmhad_down': 3,
                '3dhp_full_down': 3,
                'totalcapture': 3,
                'ikea_down': 2,
                'human4d': 1,
                'fit3d_': 2,
                'chi3d_': 1,
                'humansc3d_': 1,
                'egohumans': 6,
                'dna_rendering': 6,
            }
        elif 'subset_synth' in FLAGS.custom:
            ds_parts3d = {'3dpeople': 4, 'sailvos': 5, 'jta_down': 3, 'hspace_': 3}
        elif 'original_3d_parts' in FLAGS.custom:
            ds_parts3d = {
                'h36m_': 4,
                'muco_downscaled': 6,
                'humbi': 5,
                '3doh_down': 3,
                'agora': 3,
                'surreal': 5,
                'panoptic_': 7,
                'aist_': 6,
                'aspset_': 4,
                'gpa_': 4,
                '3dpeople': 4,
                'sailvos': 5,
                'bml_movi': 5,
                'mads_down': 2,
                'umpm_down': 2,
                'bmhad_down': 3,
                '3dhp_full_down': 3,
                'totalcapture': 3,
                'jta_down': 3,
                'ikea_down': 2,
                'human4d': 1,
                'behave_down': 3,
                'rich_down': 4,
                'spec_down': 2,
                'fit3d_': 2,
                'chi3d_': 1,
                'humansc3d_': 1,
                'hspace_': 3,
            }
        else:
            ds_parts3d = {
                'h36m_': 10,
                'muco_downscaled': 6,
                'humbi': 5,
                '3doh_down': 3,
                'panoptic_': 7,
                'aist_': 6,
                'aspset_': 4,
                'gpa_': 4,
                '3dpeople': 4,
                'sailvos': 5,
                'bml_movi': 5,
                'mads_down': 2,
                'umpm_down': 2,
                'bmhad_down': 3,
                '3dhp_full_down': 3,
                'totalcapture': 3,
                'jta_down': 3,
                'ikea_down': 2,
                'human4d': 1,
                'fit3d_': 2,
                'chi3d_': 1,
                'humansc3d_': 1,
                'hspace_': 3,
                'egohumans': 6,
                'dna_rendering': 6,
            }
        return ds_parts3d

    def get_parts_param(self):
        if 'finetune_3dpw' in FLAGS.custom:
            ds_parts_param = dict(
                tdpw=2 * 97,
                agora=8,
                bedlam=30,
                rich=6,
                behave=5,
                spec=4,
                surreal=8,
                moyo=8,
                arctic=5,
                intercap=5,
                genebody=0,
                egobody=3,
                hi4d_down=3,
                hi4d_rerender=5,
                humman=2,
                synbody_humannerf=5,
                thuman2=6,
                zjumocap=3,
                dfaust_render=3,
            )
        elif 'finetune_agora' in FLAGS.custom:
            ds_parts_param = dict(
                agora=8 + 2 * 97,
                bedlam=30,
                rich=6,
                behave=5,
                spec=4,
                surreal=8,
                moyo=8,
                arctic=5,
                intercap=5,
                genebody=0,
                egobody=3,
                hi4d_down=3,
                hi4d_rerender=5,
                humman=2,
                synbody_humannerf=5,
                thuman2=6,
                zjumocap=3,
                dfaust_render=3,
            )
        elif 'subset_real' in FLAGS.custom:
            ds_parts_param = dict(
                rich=5,
                behave=5,
                moyo=8,
                arctic=3,
                intercap=3,
                genebody=1,
                egobody=1,
                hi4d_down=3,
                humman=2,
                zjumocap=1,
            )
        elif 'subset_synth' in FLAGS.custom:
            ds_parts_param = dict(
                agora=10,
                bedlam=16,
                spec=3,
                surreal=8,
                hi4d_rerender=8,
                synbody_humannerf=6,
                thuman2=6,
                dfaust_render=8,
            )
        else:
            ds_parts_param = dict(
                agora=10,
                bedlam=16,
                rich=5,
                behave=5,
                spec=3,
                surreal=8,
                moyo=8,
                arctic=3,
                intercap=3,
                genebody=1,
                egobody=1,
                hi4d_down=3,
                hi4d_rerender=8,
                humman=2,
                synbody_humannerf=6,
                thuman2=6,
                zjumocap=1,
                dfaust_render=8,
            )
        if 'no_egobody' in FLAGS.custom:
            ds_parts_param['egobody'] = 0
        return ds_parts_param

    def render(self):
        class DummyTrainer(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

        trainer = DummyTrainer(self.model).to(torch.device('cuda'))
        callback = render_callback.RenderPredictionCallback(start_step=0, interval=1)
        callback.device = torch.device('cuda')
        callback.trainer = trainer
        step = self._n_completed_steps_at_start
        callback.on_train_begin(step)
        callback.on_train_batch_end(step, {})
        callback.on_train_end(step)

        import cv2
        import imageio.v2 as imageio
        import more_itertools
        import cameralib
        import os.path as osp
        from nlf.rendering import Renderer
        from nlf.pt.render_callback import make_triplet
        from nlf.pt.loading.common import make_marker

        image_dir = f'/work/sarandi/pexels_crops/'
        image_paths = spu.sorted_recursive_glob(f'{image_dir}/*.*')
        image_dir2 = f'/work/sarandi/hard_poses/elastic_crop'
        image_paths2 = spu.sorted_recursive_glob(f'{image_dir2}/*.*')
        image_paths = image_paths + image_paths2
        camera = cameralib.Camera.from_fov(30, [FLAGS.proc_side, FLAGS.proc_side])

        canonical_points = torch.tensor(
            np.load(f'{PROJDIR}/canonical_vertices_smplx.npy'), dtype=torch.float32
        ).cuda()
        intrinsics = (
            torch.tensor(camera.intrinsic_matrix, dtype=torch.float32).unsqueeze(0)
        ).cuda()
        camera.scale_output(512 / FLAGS.proc_side)
        faces = np.load(f'{PROJDIR}/smplx_faces.npy')
        renderer = Renderer(imshape=(512, 512), faces=faces)

        for image_path_b in more_itertools.chunked(image_paths, 4):
            image_stack_np = np.stack(
                [
                    cv2.resize(imageio.imread(p)[..., :3], (FLAGS.proc_side, FLAGS.proc_side))
                    for p in image_path_b
                ],
                axis=0,
            )

            # start = FLAGS.proc_side // 2 - 7
            # end = start + 14
            # marker = make_marker(14)
            # for i in range(len(image_stack_np)):
            #     image_stack_np[i, start:end, start:end, :] = marker

            image_stack_torch = torch.tensor(image_stack_np, dtype=torch.float32).permute(
                0, 3, 1, 2
            ).contiguous().cuda() / np.float32(255.0)

            with torch.inference_mode(), torch.amp.autocast(
                'cuda', dtype=torch.float16
            ), torch.device('cuda'):
                trainer.eval()
                pred_vertices, uncerts = trainer.model.predict_multi_same_canonicals(
                    image_stack_torch, intrinsics.repeat(len(image_path_b), 1, 1), canonical_points
                )
                pred_vertices = pred_vertices.cpu().numpy() / 1000

            image_stack = np.array(
                [
                    cv2.resize(im, (512, 512), interpolation=cv2.INTER_CUBIC)
                    for im in image_stack_np
                ]
            )
            os.makedirs(f'{self.logdir}/pred_{step:07d}', exist_ok=True)
            for im, verts, impath in zip(image_stack, pred_vertices, image_path_b):
                grid = make_triplet(im, verts, renderer, camera)
                path = f'{self.logdir}/pred_{step:07d}/{osp.basename(impath)}.jpg'
                imageio.imwrite(path, grid, quality=93)


def get_examples(dataset, learning_phase):
    if learning_phase == TRAIN:
        str_example_phase = FLAGS.train_on
    elif learning_phase == VALID:
        str_example_phase = FLAGS.validate_on
    elif learning_phase == TEST:
        str_example_phase = FLAGS.test_on
    else:
        raise Exception(f'No such learning_phase as {learning_phase}')

    if str_example_phase == 'train':
        examples = dataset.examples[TRAIN]
    elif str_example_phase == 'valid':
        examples = dataset.examples[VALID]
    elif str_example_phase == 'test':
        examples = dataset.examples[TEST]
    elif str_example_phase == 'trainval':
        examples = dataset.examples[TRAIN] + dataset.examples[VALID]
    else:
        raise Exception(f'No such phase as {str_example_phase}')
    return examples


def exp_decay(step, start_step, initial_value, decay_rate, decay_steps):
    return initial_value * decay_rate ** ((step - start_step) / decay_steps)


def cos_decay(step, start_step, initial_value, final_value, decay_steps):
    return (
        (initial_value - final_value)
        * 0.5
        * (1 + np.cos((step - start_step) / decay_steps * np.pi))
    )


def get_sections_and_sizes(examples, section_name_to_size, verify_all_included=False):
    section_names, section_sizes = zip(*section_name_to_size.items())
    section_names_sorted = sorted(section_names)

    sections_sorted = [[] for _ in section_names_sorted]
    i = 0
    for ex in spu.progressbar(examples, desc='Building dataset sections'):
        if i >= 0 and ex.image_path.startswith(section_names_sorted[i]):
            sections_sorted[i].append(ex)
        else:
            i = bisect.bisect(section_names_sorted, ex.image_path) - 1
            if i >= 0 and ex.image_path.startswith(section_names_sorted[i]):
                sections_sorted[i].append(ex)
            elif verify_all_included:
                raise RuntimeError(f'No section for {ex.image_path}')

    if not all(len(s) > 0 for s in sections_sorted):
        for name, s in zip(section_names_sorted, sections_sorted):
            print(f'{name}: {len(s)}')
        raise RuntimeError('Some sections are empty')

    sections_name_to_section = dict(zip(section_names_sorted, sections_sorted))
    sections = [sections_name_to_section[name] for name in section_names]
    return sections, section_sizes


class AdjustRenormClipping(florch.callbacks.Callback):
    def __init__(self, ramp_start_step, ramp_length):
        super().__init__()
        self.ramp_start_step = ramp_start_step
        self.ramp_length = ramp_length

    def on_train_batch_begin(self, step, logs=None):
        ramp = ptu.ramp_function(step, self.ramp_start_step, self.ramp_length)
        rmax = 1 + ramp * 2 * FLAGS.renorm_limit_scale  # ramps from 1 to 3
        dmax = ramp * 5 * FLAGS.renorm_limit_scale  # ramps from 0 to 5

        def set_renorm_clipping(module):
            if hasattr(module, 'rmax') and hasattr(module, 'rmin') and hasattr(module, 'dmax'):
                module.rmax = rmax
                module.rmin = 1.0 / rmax
                module.dmax = dmax

        self.trainer.model.backbone.apply(set_renorm_clipping)


if __name__ == '__main__':
    main()
