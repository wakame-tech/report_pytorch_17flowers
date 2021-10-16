import hydra
from pathlib import Path
from dataclasses import dataclass
from omegaconf import DictConfig


@dataclass
class Parameters:
    tgz_path: Path
    pickle_path: Path


def read_parameters(config: DictConfig) -> Parameters:
    tgz_path = Path(hydra.utils.get_original_cwd()) / config['dirs']['dataset']['tgz']
    pickle_path = Path(hydra.utils.get_original_cwd()) / config['dirs']['dataset']['pickle']
    return Parameters(
        tgz_path=tgz_path,
        pickle_path=pickle_path,
    )
