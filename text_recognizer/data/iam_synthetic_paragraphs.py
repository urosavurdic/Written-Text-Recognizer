from typing import Any, List, Sequence, Tuple
import random
from PIL import Image
import numpy as np

from text_recognizer.data.iam_paragraphs import (
    IAMParagraphs,
    get_dataset_properties,
    resize_image,
    get_transform,
    NEW_LINE_TOKEN,
    IMAGE_SCALE_FACTOR,
)
from text_recognizer.data.iam import IAM
from text_recognizer.data.iam_lines import line_crops_and_labels, save_images_and_labels, load_line_crops_and_labels
from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.util import BaseDataset, convert_str_to_labels

PROCESSED_DATA_DIRNAME = BaseDataModule.data_directory_path() / "processed" / "iam_synthetic_paragraphs"

class IAMSyntheticParagraphs(IAMParagraphs):
    """
    DataModule for synthetic paragraphs generated from IAM line images.
    Each paragraph is created by randomly combining line images from the IAM Lines dataset.
    This is useful for training and evaluating text recognition models on paragraph-level data and simulating real-world scenarios.
    Used only for training, not for validation or testing.
    """
    
    def prepare_data(self, *args, **kwargs) -> None:
        """
        Prepares the synthetic paragraph dataset by generating paragraphs from IAM line crops such that they can be used to generate synthetic paragraphs dataset in setup().
        This method is IAMLines.prepare_data + resizing of line crops.
        """
        if PROCESSED_DATA_DIRNAME.exists():
            return
        print("IAMSyntheticParagraphs.prepare_data: preparing IAM lines for synthetic IAM paragraph creation...")
        print("Cropping IAM line regions and loading labels...")
        iam = IAM()
        iam.prepare_data()
        # generate line
        crops_train_val, labels_train_val = line_crops_and_labels(iam, "train_val")
        crops_test, labels_test = line_crops_and_labels(iam, "test")
        # resize line crops
        crops_train_val = [resize_image(crop, IMAGE_SCALE_FACTOR) for crop in crops_train_val] 
        crops_test = [resize_image(crop, IMAGE_SCALE_FACTOR) for crop in crops_test]

        print(f"Saving images and labels at {PROCESSED_DATA_DIRNAME}...")
        save_images_and_labels(crops_train_val, labels_train_val, "train_val", PROCESSED_DATA_DIRNAME)
        save_images_and_labels(crops_test, labels_test, "test", PROCESSED_DATA_DIRNAME)

    def setup(self, stage: str = None) -> None:
        print(f"IAMSyntheticParagraphs.setup({stage}): Loading trainval IAM paragraph regions and lines...")

        if stage == "fit" or stage is None:
            line_crops, line_labels = load_line_crops_and_labels("train_val", PROCESSED_DATA_DIRNAME)
            X, para_labels = generate_synthetic_paragraphs(line_crops=line_crops, line_labels=line_labels)
            Y = convert_str_to_labels(strings=para_labels, mapping=self.idx_to_char, length=self.output_dim[0])
            transform = get_transform(image_shape=self.dim[1:], augment=self.augment)
            self.data_train = BaseDataset(X, Y, transform=transform)

    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "IAM Synthetic Paragraphs Dataset\n"
            f"Num classes: {len(self.char_to_idx)}\n"
            f"Input dims : {self.dim}\n"
            f"Output dims: {self.output_dim}\n"
        )
        if self.data_train is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, 0, 0\n"
            f"Train Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Train Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        return basic + data

def generate_synthetic_paragraphs(
        line_crops: List[Image.Image], line_labels: List[str], max_batch_size: int = 9
        ) -> Tuple[List[Image.Image], List[str]]:
    """
    Generates synthetic paragraphs by randomly combining line crops and their corresponding labels.
    Args:
        line_crops (List[Image.Image]): List of line crop images.
        line_labels (List[str]): List of corresponding line labels.
        max_batch_size (int): Maximum number of lines to combine into a single paragraph.
    Returns:
        Tuple[List[Image.Image], List[str]]: A tuple containing a list of synthetic paragraph images and their corresponding labels.
    """

    paragraph_properties = get_dataset_properties()
    indices = list(range(len(line_labels)))
    assert max_batch_size < paragraph_properties["num_lines"]["max"]

    # to ensure that we have at least one paragraph with max_batch_size lines and one with min_batch_size lines in the dataset
    batched_indices_list = [[_] for _ in indices]
    batched_indices_list.extend(
        generate_random_batches(values=indices, min_batch_size=2, max_batch_size=max_batch_size // 2)
    )
    batched_indices_list.extend(
        generate_random_batches(values=indices, min_batch_size=2, max_batch_size=max_batch_size)
    )
    batched_indices_list.extend(
        generate_random_batches(values=indices, min_batch_size=(max_batch_size // 2) + 1, max_batch_size=max_batch_size)
    )
    # print some stats
    unique, counts = np.unique([len(_) for _ in batched_indices_list], return_counts=True)
    for batch_len, count in zip(unique, counts):
        print(f"{count} samples with {batch_len} lines")

    para_crops, para_labels = [], []
    for para_indices in batched_indices_list:
        # create paragraph label and crop
        para_label = NEW_LINE_TOKEN.join([line_labels[i] for i in para_indices])
        if len(para_label) > paragraph_properties["label_length"]["max"]:
            print("Label longer than longest label in original IAM Paragraphs dataset - hence dropping")
            continue

        para_crop = join_line_crops_to_form_paragraph([line_crops[i] for i in para_indices])
        max_para_shape = paragraph_properties["crop_shape"]["max"]
        if para_crop.height > max_para_shape[0] or para_crop.width > max_para_shape[1]:
            print("Crop larger than largest crop in original IAM Paragraphs dataset - hence dropping")
            continue
        # append to lists
        para_crops.append(para_crop)
        para_labels.append(para_label)

    assert len(para_crops) == len(para_labels)

    return para_crops, para_labels


def join_line_crops_to_form_paragraph(line_crops: Sequence[Image.Image]) -> Image.Image:
    """
    Horizontally stacks line crops to form a paragraph crop.
    Args:
        line_crops (Sequence[Image.Image]): Sequence of line crop images.
    Returns:
        Image.Image: The resulting paragraph crop image.
    """
    crop_shapes = np.array([_.size[::-1] for _ in line_crops])
    para_height = crop_shapes[:, 0].sum()
    para_width = crop_shapes[:, 1].max()

    para_image = Image.new(mode="L", size=(para_width, para_height), color=0)
    current_height = 0
    for line_crop in line_crops:
        para_image.paste(line_crop, box=(0, current_height))
        current_height += line_crop.height
    return para_image

def generate_random_batches(values: List[Any], min_batch_size: int, max_batch_size: int) -> List[List[Any]]:
    """
    Generates random batches of values with sizes between min_batch_size and max_batch_size.
    Args:
        values (List[Any]): List of values to be batched.
        min_batch_size (int): Minimum size of each batch.
        max_batch_size (int): Maximum size of each batch.
    Returns:
        List[List[Any]]: A list of batches, where each batch is a list of values.
    """
    shuffled_values = values.copy()
    random.shuffle(shuffled_values)

    start_id = 0
    grouped_values_list = []
    while start_id < len(shuffled_values):
        num_values = random.randint(min_batch_size, max_batch_size)
        grouped_values_list.append(shuffled_values[start_id : start_id + num_values])
        start_id += num_values
    assert sum([len(_) for _ in grouped_values_list]) == len(values)
    
    return grouped_values_list


if __name__ == "__main__":
    load_and_print_info(IAMSyntheticParagraphs)