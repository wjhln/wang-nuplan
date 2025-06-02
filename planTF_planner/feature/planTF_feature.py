from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
)


def to_tensor(data):
    if isinstance(data, dict):
        return to_tensor(data)
    elif isinstance(data, np.ndarray):
        if data.dtype is bool:
            return torch.from_numpy(data).bool
        elif data.dtype is float:
            return torch.from_numpy(data).float
    elif isinstance(data, np.number):
        return torch.tensor(data).float
    else:
        print(type(data), data)
        raise NotImplementedError


def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        raise NotImplementedError


@dataclass
class PlanTFFeature(AbstractModelFeature):
    data: Dict[str, Any]

    @classmethod
    def collate(cls, feature_list: List[PlanTFFeature]) -> PlanTFFeature:
        batch_data = {}
        for key in ["agent", "map"]:
            batch_data[key] = {
                k: pad_sequence(
                    [f.data[key][k] for f in feature_list], batch_first=True
                )
                for k in feature_list[0].data[key].keys()
            }

        return PlanTFFeature(batch_data)

    def to_feature_tensor(self) -> PlanTFFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_tensor(v)
        return PlanTFFeature(new_data)

    def to_device(self, device: torch.device) -> PlanTFFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_device(v, device)
        return PlanTFFeature(new_data)

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> PlanTFFeature:
        return PlanTFFeature(data)

    def unpack(self) -> List[PlanTFFeature]:
        raise NotImplementedError

    # 全局坐标转为自车坐标
    def normalize(self, data, first_time=False, radius=0, hist_step=21):
        cur_state = data["current_state"]
        center_xy, center_angle = cur_state[:2].copy(), cur_state[2].copy()
        rotate_mat = np.array(
            [
                [np.cos(center_angle), -np.sin(center_angle)],
                [np.sin(center_angle), np.cos(center_angle)],
            ],
            dtype=np.float64,
        )

        data["current_state"][:3] = 0
        data["agent"]["position"] = np.matmul(
            data["agent"]["position"] - center_xy, rotate_mat
        )
        data["agent"]["velocity"] = np.matmul(data["agent"]["velocity"], rotate_mat)
        data["agent"]["heading"] -= center_angle

        data["map"]["point_position"] = np.matmul(
            data["map"]["point_position"] - center_xy, rotate_mat
        )
        data["map"]["point_vector"] = np.matmul(data["map"]["point_vector"], rotate_mat)
        data["map"]["point_orientation"] -= center_angle

        data["map"]["polygon_center"][..., :2] = np.matmul(
            data["map"]["polygon_center"][..., :2] - center_xy, rotate_mat
        )
        data["map"]["polygon_orientation"] -= center_angle

        target_position = (
            data["agent"]["position"][:, hist_step:]
            - data["agent"]["position"][:, hist_step - 1][:, None]
        )
        target_heading = (
            data["agent"]["heading"][:, hist_step:]
            - data["agent"]["heading"][:, hist_step - 1][:, None]
        )
        target = np.concatenate([target_position, target_heading[..., None]], axis=-1)
        target[~data["agent"]["valid_mask"][:, hist_step:]] = 0
        data["agent"]["target"] = target

        if first_time:
            point_position = data["map"]["point_position"]
            x_max, x_min = radius, -radius
            y_max, y_min = radius, -radius
            valid_mask = (
                (point_position[:, 0, :, 0] < x_max)
                & (point_position[:, 0, :, 0] > x_min)
                & (point_position[:, 0, :, 1] < y_max)
                & (point_position[:, 0, :, 1] > y_min)
            )  # [M, P]
            data["map"]["valid_mask"] = valid_mask
            valid_polygon = valid_mask.any(-1)  # [M]

            for k, v in data["map"].items():
                data["map"][k] = v[valid_polygon]

            data["origin"] = center_xy
            data["angle"] = center_angle
        return PlanTFFeature(data)
