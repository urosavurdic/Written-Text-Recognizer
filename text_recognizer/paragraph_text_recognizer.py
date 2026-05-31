from pathlib import Path
from typing import Sequence, Union
import argparse
import json

from PIL import Image
import torch

from text_recognizer.data import IAMParagraphs
from text_recognizer.data.iam_paragraphs import resize_image, IMAGE_SCALE_FACTOR, get_transform
from text_recognizer.lit_models import TransformerModel
from text_recognizer.models import ResnetTransformer
import text_recognizer.util as util


CONFIG_AND_WEIGHTS_DIRNAME = Path(__file__).resolve().parent / "artifacts" / "paragraph_text_recognizer"


class ParagraphTextRecognizer:
    def __init__(self):
        data = IAMParagraphs()
        self.char_to_idx = data.char_to_idx
        self.idx_to_char = data.idx_to_char
        self.mapping = [self.idx_to_char[i] for i in range(len(self.idx_to_char))]
        self.ignore_tokens = [
            self.char_to_idx["<S>"],
            self.char_to_idx["<B>"],
            self.char_to_idx["<E>"],
            self.char_to_idx["<P>"],
        ]
        self.transform = get_transform(image_shape=data.dim[1:], augment=False)

        with open(CONFIG_AND_WEIGHTS_DIRNAME / "config.json", "r") as file:
            config = json.load(file)
        args = argparse.Namespace(**config)

        model = ResnetTransformer(data_config=data.configuration(), args=args)
        self.lit_model = TransformerModel.load_from_checkpoint(
            checkpoint_path=CONFIG_AND_WEIGHTS_DIRNAME / "model.pt", args=args, model=model
        )
        self.lit_model.eval()
        self.scripted_model = self.lit_model.to_torchscript(method="script", file_path=None)
    
    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image]) -> str:
        """
        Predict/infer text in input image (which can be a file path).
        """
        image_pil = image
        if not isinstance(image, Image.Image):
            image_pil = util.read_image_pil(image, grayscale=True)

        image_pil = resize_image(image_pil, IMAGE_SCALE_FACTOR)
        image_tensor = self.transform(image_pil)

        y_pred = self.scripted_model(image_tensor.unsqueeze(axis=0))[0]
        pred_str = convert_y_label_to_string(y=y_pred, mapping=self.mapping, ignore_tokens=self.ignore_tokens)

        return pred_str
    
def convert_y_label_to_string(y: torch.Tensor, mapping: Sequence[str], ignore_tokens: Sequence[int]) -> str:
    return "".join([mapping[i] for i in y if i not in ignore_tokens])

def main():
    """
    Example runs:
    python text_recognizer/paragraph_text_recognizer.py text_recognizer/tests/support/paragraphs/a01-077.png
    python text_recognizer/paragraph_text_recognizer.py https://fsdl-public-assets.s3-us-west-2.amazonaws.com/paragraphs/a01-077.png
    """
    parser = argparse.ArgumentParser(description="Recognize handwritten text in an image file.")
    parser.add_argument("filename", type=str)
    args = parser.parse_args()

    text_recognizer = ParagraphTextRecognizer()
    pred_str = text_recognizer.predict(args.filename)
    print(pred_str)


if __name__ == "__main__":
    main()

