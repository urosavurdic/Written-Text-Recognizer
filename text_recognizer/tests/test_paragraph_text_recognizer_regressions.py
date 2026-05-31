from PIL import Image
import torch

from text_recognizer import paragraph_text_recognizer as paragraph_module


def test_paragraph_text_recognizer_mapping_and_prediction(monkeypatch):
    class DummyData:
        char_to_idx = {"<S>": 0, "<B>": 1, "<E>": 2, "<P>": 3, "a": 4}
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        dim = (1, 8, 8)

        @staticmethod
        def configuration():
            return {}

    class DummyLitModel:
        def eval(self):
            return None

        @staticmethod
        def to_torchscript(method="script", file_path=None):
            return lambda _x: [torch.tensor([4, 2, 3])]

    class DummyTransformerModel:
        @staticmethod
        def load_from_checkpoint(checkpoint_path, args, model):
            return DummyLitModel()

    monkeypatch.setattr(paragraph_module, "IAMParagraphs", lambda: DummyData())
    monkeypatch.setattr(paragraph_module, "get_transform", lambda image_shape, augment: lambda _: torch.zeros((1, 8, 8)))
    monkeypatch.setattr(paragraph_module, "ResnetTransformer", lambda data_config, args: object())
    monkeypatch.setattr(paragraph_module, "TransformerModel", DummyTransformerModel)
    monkeypatch.setattr(paragraph_module, "resize_image", lambda image, scale_factor: image)
    monkeypatch.setattr(paragraph_module.util, "read_image_pil", lambda *_args, **_kwargs: Image.new("L", (8, 8)))

    recognizer = paragraph_module.ParagraphTextRecognizer()
    assert recognizer.mapping == ["<S>", "<B>", "<E>", "<P>", "a"]
    assert recognizer.ignore_tokens == [0, 1, 2, 3]
    assert recognizer.predict("fake-path.png") == "a"
