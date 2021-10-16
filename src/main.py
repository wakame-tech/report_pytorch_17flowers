import torch
from pytorch_lightning.trainer import Trainer
import hydra
from omegaconf import DictConfig
import logging

from logger import prepare_logger, set_logger
from parameters import read_parameters
from model import Model


@hydra.main(config_path='..', config_name='config')
def main(config: DictConfig):
    logger = prepare_logger(logging.DEBUG)
    set_logger(logger)
    params = read_parameters(config)

    # TODO:
    device = torch.device('cpu')
    model = Model(params)
    model = model.to(device)

    trainer = Trainer()
    trainer.fit(model)

if __name__ == '__main__':
    main()
