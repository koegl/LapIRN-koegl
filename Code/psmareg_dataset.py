from pathlib import Path
from typing import Dict

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


def _minmax_norm(x: np.ndarray) -> np.ndarray:
    min_v = np.min(x)
    max_v = np.max(x)
    return (x - min_v) / (max_v - min_v + 1e-8)


class PSMARegPairDataset(Dataset):
    def __init__(
        self,
        fixed_ct_path: Path,
        fixed_pet_path: Path,
        moving_ct_path: Path,
        moving_pet_path: Path,
    ) -> None:
        super().__init__()
        self.fixed_ct_path = fixed_ct_path
        self.fixed_pet_path = fixed_pet_path
        self.moving_ct_path = moving_ct_path
        self.moving_pet_path = moving_pet_path
        
        self.fixed_ct = self._load_volume(fixed_ct_path)
        self.fixed_pet = self._load_volume(fixed_pet_path)
        self.moving_ct = self._load_volume(moving_ct_path)
        self.moving_pet = self._load_volume(moving_pet_path)

    def _load_volume(self, path: Path) -> torch.Tensor:
        volume = nib.load(str(path)).get_fdata()
        volume = np.asarray(volume, dtype=np.float32)
        volume = _minmax_norm(volume)
        volume = np.reshape(volume, (1,) + volume.shape)
        return torch.from_numpy(volume.astype(np.float32))

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "fixed_ct": self.fixed_ct,
            "fixed_pet": self.fixed_pet,
            "moving_ct": self.moving_ct,
            "moving_pet": self.moving_pet,
        }
