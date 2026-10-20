"""Stage-transition contracts, runnable without torch/MMCV/MMEngine.

These checks execute each Python configuration's local assignments. They do not
claim to build the inherited MMEngine config or run training.
"""
import os
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / 'projects' / 'OCRDINO' / 'configs'
PRETRAIN = CONFIGS / 'ocrdino_r50_4scale_query_pretrain_24e.py'
FINETUNE = CONFIGS / 'ocrdino_r50_4scale_query_totaltext_12e.py'


class TestTrainingStages(unittest.TestCase):
    def load(self, path, env=None):
        with patch.dict(os.environ, env or {}, clear=True):
            return runpy.run_path(str(path))

    def test_old_generic_stage_variables_cannot_change_the_requested_plan(self):
        old_env = dict(OCRDINO_EPOCHS='99', OCRDINO_LR='0.9',
                       OCRDINO_TRAIN_JSON='annotations/totaltext_test.json',
                       OCRDINO_WORK_DIR='work_dirs/old_run')
        cases = [(PRETRAIN, 24, 1e-4, [19, 22], 'pretrain'),
                 (FINETUNE, 12, 5e-5, [8, 11], 'totaltext_train')]
        for path, epochs, lr, milestones, dataset in cases:
            with self.subTest(path=path.name):
                cfg = self.load(path, old_env)
                self.assertEqual(cfg['train_cfg']['max_epochs'], epochs)
                self.assertGreater(cfg['train_cfg']['val_interval'], epochs)
                self.assertEqual(cfg['optim_wrapper']['optimizer']['lr'], lr)
                schedule = cfg['param_scheduler'][1]
                self.assertEqual(schedule['end'], epochs)
                self.assertEqual(schedule['milestones'], milestones)
                self.assertEqual(cfg['train_dataloader']['dataset']['ann_file'],
                                 'annotations/' + dataset + '.json')
                self.assertNotEqual(cfg['work_dir'], old_env['OCRDINO_WORK_DIR'])

    def test_finetune_loads_full_ocr_checkpoint_without_resuming_coco_init(self):
        env = dict(OCRDINO_PRETRAINED='/checkpoints/converted_coco.pth')
        pretrain = self.load(PRETRAIN, env)
        finetune = self.load(FINETUNE, env)
        self.assertEqual(pretrain['load_from'], env['OCRDINO_PRETRAINED'])
        self.assertEqual(finetune['load_from'],
                         'work_dirs/ocrdino_query_pretrain24/epoch_24.pth')
        self.assertIs(pretrain['resume'], False)
        self.assertIs(finetune['resume'], False)
        self.assertNotEqual(pretrain['work_dir'], finetune['work_dir'])
        selected = self.load(FINETUNE, dict(
            env, OCRDINO_FINETUNE_FROM='/runs/pretrain/epoch_24.pth'))
        self.assertEqual(selected['load_from'], '/runs/pretrain/epoch_24.pth')

    def test_both_stages_use_the_same_model_and_dataset_namespace(self):
        for path in (PRETRAIN, FINETUNE):
            with self.subTest(path=path.name):
                cfg = self.load(path)
                base = path.parent / cfg['_base_']
                self.assertEqual(base.resolve(),
                                 (CONFIGS / 'ocrdino_r50_4scale_query.py').resolve())
                self.assertTrue(base.is_file())
                self.assertNotIn('model', cfg)
                dataset = cfg['train_dataloader']['dataset']
                self.assertEqual(dataset['data_root'], 'data/english')
                self.assertEqual(dataset['data_prefix']['img'], '')
                cfg = self.load(path, dict(
                    OCRDINO_DATA_ROOT='/datasets/english',
                    OCRDINO_IMAGES='/mnt/extracted'))
                dataset = cfg['train_dataloader']['dataset']
                self.assertEqual(dataset['data_root'], '/datasets/english')
                self.assertEqual(dataset['data_prefix']['img'], '/mnt/extracted')


if __name__ == '__main__':
    unittest.main()



def test_mmengine_resolves_official_dino_inheritance():
    import pytest
    Config = pytest.importorskip('mmengine').Config
    paths = [CONFIGS / 'ocrdino_r50_4scale_query.py', PRETRAIN, FINETUNE]
    for path in paths:
        with patch.dict(os.environ, {}, clear=True):
            cfg = Config.fromfile(str(path), import_custom_modules=False)
        assert cfg.model.type == 'OCRDINO'
        assert cfg.model.backbone.type == 'ResNet'
        assert cfg.model.backbone.depth == 50
        assert tuple(cfg.model.backbone.out_indices) == (0, 1, 2, 3)
        assert cfg.model.backbone.with_cp is True
        assert cfg.model.num_queries == 900
        assert cfg.model.neck.num_outs == 4
        assert cfg.model.encoder.num_layers == 6
        assert cfg.model.decoder.num_layers == 6
        rec = cfg.model.recognition_head
        assert rec.type == 'CharacterQueryHead'
        assert rec.embed_dims == 256 and rec.num_layers == 3
        assert rec.num_heads == 8 and rec.ffn_dim == 1024
        assert tuple(rec.query_buckets) == (32, 64, 128, 256, 500)
        assert rec.query_width_scale == 16.
        assert cfg.model.rec_train_cfg.max_rois == 128
        assert cfg.model.rec_test_cfg.max_per_img == 100
        assert cfg.train_dataloader.batch_size == 1
        assert cfg.optim_wrapper.accumulative_counts == 4
        assert cfg.optim_wrapper.type == 'AmpOptimWrapper'
        assert cfg.val_cfg is None and cfg.val_dataloader is None
        assert cfg.resume is False


def test_legacy_configs_require_explicit_migration():
    import pytest
    for name in ['ocrdino_r50_4scale_ctc.py',
                 'ocrdino_r50_4scale_ctc_pretrain_24e.py',
                 'ocrdino_r50_4scale_ctc_totaltext_12e.py']:
        with pytest.raises(RuntimeError, match='query'):
            runpy.run_path(str(CONFIGS / name))
