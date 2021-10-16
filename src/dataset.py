"""
データセットの前処理関係
"""
import re
from typing import Union, Tuple, List
import tarfile
from pathlib import Path

import torch.utils.data

from logger import get_logger
from parameters import Parameters
from glob import glob
from PIL import Image
from tqdm import tqdm
import pickle

# データセットの型
Dataset = List[Tuple[Image.Image, int]]

def extract_dataset_tgz(params: Parameters) -> Union[Path, None]:
    logger = get_logger()
    logger.info('- extract tar...')

    images_path = Path(f'{params.tgz_path.parent}/jpg')
    if images_path.exists():
        logger.info('-> skip')
        return images_path

    if not params.tgz_path.exists():
        logger.error('flower17 dataset not found')
        return None
    tar = tarfile.open(params.tgz_path)
    tar.extractall(params.tgz_path.parent)
    tar.close()

    logger.info(f'-> completed @{images_path}')
    return images_path


def get_image_label(path: Path) -> Union[int, None]:
    match = re.match(r'image_(\d+).jpg', path.name)
    if match is None:
        return None
    num = int(match.group(1))
    if not (1 <= num <= 1360):
        return None
    return (num - 1) // 80
    # return [
    #     'Tulip',
    #     'Snowdrop',
    #     'LilyValley',
    #     'Bluebell',
    #     'Crocus',
    #     'Iris',
    #     'Tigerlily',
    #     'Daffodil',
    #     'Fritillary',
    #     'Sunflower',
    #     'Daisy',
    #     'ColtsFoot',
    #     'Dandelion',
    #     'Cowslip',
    #     'Buttercup',
    #     'Windflower',
    #     'Pansy',
    # ][(num - 1) // 80]


def process_image_labels(images_path: Path) -> Dataset:
    logger = get_logger()
    logger.info(f'- Process images')

    res = []
    for image_path_str in tqdm(glob(f'{images_path}/*.jpg')):
        image_path = Path(image_path_str)
        label = get_image_label(image_path)
        if label is None:
            continue
        # logger.debug(f'{image_path_str} -> {label}')
        im = Image.open(image_path)
        im = im.resize((256, 256))
        res.append((im, label))

    logger.info(f'{len(res)} images processed')
    return res


def save_images_and_labels_as_pickle(images_and_labels: Dataset, pickle_path: Path):
    logger = get_logger()
    logger.info(f'- Save as Pickle')

    with open(pickle_path, 'wb') as pkl:
        pickle.dump(images_and_labels, file=pkl)

    logger.info(f'dump as pickle @{pickle_path}')


def load_images_and_labels_as_pickle(pickle_path: Path) -> Union[Dataset, None]:
    logger = get_logger()
    logger.info(f'- Load Pickle')
    with open(pickle_path, 'rb') as pkl:
        images_and_labels = pickle.load(pkl)

    return images_and_labels


def load_flower17_dataset(params: Parameters) -> Union[Dataset, None]:
    # preprocess
    if not params.pickle_path.exists():
        images_path = extract_dataset_tgz(params)
        if images_path is None:
            return None
        image_and_labels = process_image_labels(images_path)
        save_images_and_labels_as_pickle(image_and_labels, params.pickle_path)

    return load_images_and_labels_as_pickle(params.pickle_path)


"""
Flower17データセット
"""
class Flower17(torch.utils.data.Dataset):
    def __init__(self, params: Parameters, transform):
        self.items = load_flower17_dataset(params)
        self.transform = transform
        self.data_counts = len(self.items)

    def __len__(self) -> int:
        return self.data_counts

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        im, label = self.items[index]
        # apply transforms
        im: torch.Tensor = self.transform(im)
        return im, label