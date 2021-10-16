"""
データセットの前処理関係
"""
import numpy as np
import re
from typing import Union, Tuple, List
import tarfile
from pathlib import Path
from logger import get_logger
from parameters import Parameters
from glob import glob
from PIL import Image
from tqdm import tqdm
import pickle


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


def get_image_label(path: Path) -> Union[str, None]:
    match = re.match(r'image_(\d+).jpg', path.name)
    if match is None:
        return None
    num = int(match.group(1))
    if not (1 <= num <= 1360):
        return None
    return [
        'Tulip',
        'Snowdrop',
        'LilyValley',
        'Bluebell',
        'Crocus',
        'Iris',
        'Tigerlily',
        'Daffodil',
        'Fritillary',
        'Sunflower',
        'Daisy',
        'ColtsFoot',
        'Dandelion',
        'Cowslip',
        'Buttercup',
        'Windflower',
        'Pansy',
    ][(num - 1) // 80]


def process_image_labels(images_path: Path) -> List[Tuple[np.ndarray, str]]:
    logger = get_logger()
    logger.info(f'- Process images')

    res = []
    for image_path_str in tqdm(glob(f'{images_path}/*.jpg')):
        image_path = Path(image_path_str)
        label = get_image_label(image_path)
        if label is None:
            continue
        # logger.debug(f'{image_path_str} -> {label}')
        im = np.array(Image.open(image_path))
        res.append((im, label))

    logger.info(f'{len(res)} images processed')
    return res


def save_images_and_labels_as_pickle(images_and_labels: List[Tuple[np.ndarray, str]], pickle_path: Path):
    logger = get_logger()
    logger.info(f'- Save as Pickle')

    with open(pickle_path, 'wb') as pkl:
        pickle.dump(images_and_labels, file=pkl)

    logger.info(f'dump as pickle @{pickle_path}')


def load_images_and_labels_as_pickle(pickle_path: Path) -> Union[List[Tuple[np.ndarray, str]], None]:
    logger = get_logger()
    logger.info(f'- Load Pickle')
    with open(pickle_path, 'rb') as pkl:
        images_and_labels = pickle.load(pkl)

    return images_and_labels


def load_dataset(params: Parameters) -> Union[List[Tuple[np.ndarray, str]], None]:
    # preprocess
    if not params.pickle_path.exists():
        images_path = extract_dataset_tgz(params)
        if images_path is None:
            return None
        image_and_labels = process_image_labels(images_path)
        save_images_and_labels_as_pickle(image_and_labels, params.pickle_path)

    return load_images_and_labels_as_pickle(params.pickle_path)