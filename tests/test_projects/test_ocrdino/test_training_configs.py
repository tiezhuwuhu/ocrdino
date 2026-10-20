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
PRETRAIN = CONFIGS / 'ocrdino_r50_4scale_ctc_pretrain_24e.py'
FINETUNE = CONFIGS / 'ocrdino_r50_4scale_ctc_totaltext_12e.py'


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
                         'work_dirs/ocrdino_pretrain24/epoch_24.pth')
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
                                 (CONFIGS / 'ocrdino_r50_4scale_ctc.py').resolve())
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
