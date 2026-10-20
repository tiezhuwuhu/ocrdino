"""Keep dictionary identity and Character Query semantics in checkpoints."""
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet.registry import HOOKS
from .vocabulary import recognition_metadata, validate_checkpoint_metadata


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

    def after_load_checkpoint(self, runner, checkpoint):
        # MMEngine Runner calls this after reading the file but BEFORE loading
        # its tensors into the model. Head buffers also guard direct state loads.
        validate_checkpoint_metadata(
            checkpoint, self._model(runner).recognition_head.dictionary)

    def before_save_checkpoint(self, runner, checkpoint):
        dictionary = self._model(runner).recognition_head.dictionary
        checkpoint.setdefault('meta', {})['ocrdino'] = recognition_metadata(dictionary)