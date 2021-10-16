import torch
import hydra
from omegaconf import DictConfig
import logging

from dataset import load_dataset
from logger import prepare_logger, set_logger
from parameters import read_parameters


@hydra.main(config_path='..', config_name='config')
def main(config: DictConfig):
    logger = prepare_logger(logging.DEBUG)
    set_logger(logger)
    params = read_parameters(config)

    dataset = load_dataset(params)
    print(f'image: {dataset[0][0].shape}, label: {dataset[0][1]}')


if __name__ == '__main__':
    main()
