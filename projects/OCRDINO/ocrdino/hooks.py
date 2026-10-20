"""Keep dictionary identity explicit in datasets and saved checkpoints."""
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet.registry import HOOKS


@HOOKS.register_module()
class OCRDictionaryHook(Hook):
    @staticmethod
    def _model(runner):
        model = runner.model
        return model.module if is_model_wrapper(model) else model

    def before_train(self, runner):
        model = self._model(runner)
        expected = tuple(model.recognition_head.dictionary.chars)
        dataset = runner.train_dataloader.dataset
        datasets = [dataset]
        while datasets:
            current = datasets.pop()
            if hasattr(current, 'datasets'):
                datasets.extend(current.datasets)
            elif hasattr(current, 'dataset'):
                datasets.append(current.dataset)
            elif not hasattr(current, 'dictionary'):
                raise TypeError('OCR dataset must expose its CharacterDictionary')
            elif tuple(current.dictionary.chars) != expected:
                raise ValueError('Dataset/model character dictionaries differ in content or order')

    def before_save_checkpoint(self, runner, checkpoint):
        model = self._model(runner)
        checkpoint.setdefault('meta', {})['ocrdino'] = dict(
            architecture='dino_c2_conv4_bilstm1_ctc',
            characters=list(model.recognition_head.dictionary.chars),
            blank=0, bbox_format='xyxy', dictionary_case_sensitive=True)
