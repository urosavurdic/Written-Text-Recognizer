import argparse
from torch.utils.data import ConcatDataset
from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.iam_paragraphs import IAMParagraphs
from text_recognizer.data.iam_synthetic_paragraphs import IAMSyntheticParagraphs


class IAMOriginalAndSyntheticParagraphs(BaseDataModule):
    """
    This class combines the original IAM paragraphs with synthetic paragraphs generated from IAM line images.
    It is useful for training and evaluating text recognition models on a more diverse set of paragraph-level data.
    """
    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)

        self.iam_paragraphs = IAMParagraphs(args)
        self.iam_syn_paragraphs = IAMSyntheticParagraphs(args)

        self.dim = self.iam_paragraphs.dim
        self.output_dim = self.iam_paragraphs.output_dim
        self.char_to_idx = self.iam_paragraphs.char_to_idx

        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

    @staticmethod
    def add_arguments(parser):
        BaseDataModule.add_arguments(parser)
        parser.add_argument("--augment_data", type=str, default="true")
        return parser

    def prepare_data(self, *args, **kwargs) -> None:
        self.iam_paragraphs.prepare_data()
        self.iam_syn_paragraphs.prepare_data()

    def setup(self, stage: str = None) -> None:
        self.iam_paragraphs.setup(stage)
        self.iam_syn_paragraphs.setup(stage)

        self.data_train = ConcatDataset([self.iam_paragraphs.data_train, self.iam_syn_paragraphs.data_train])
        self.data_val = self.iam_paragraphs.data_val
        self.data_test = self.iam_paragraphs.data_test
    
    def __repr__(self) -> str:
        """
        Print info about the dataset.
        """
        basic = (
            "IAM Original and Synthetic Paragraphs Dataset\n"
            f"Num classes: {len(self.char_to_idx)}\n"
            f"Dims: {self.dim}\n"
            f"Output dims: {self.output_dim}\n"
        )
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        xt, yt = next(iter(self.test_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Train Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Train Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
            f"Test Batch x stats: {(xt.shape, xt.dtype, xt.min(), xt.mean(), xt.std(), xt.max())}\n"
            f"Test Batch y stats: {(yt.shape, yt.dtype, yt.min(), yt.max())}\n"
        )
        return basic + data


if __name__ == "__main__":
    load_and_print_info(IAMOriginalAndSyntheticParagraphs)