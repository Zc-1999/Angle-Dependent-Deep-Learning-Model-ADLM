from typing import List

import numpy as np
import pandas as pd
from torch.utils.data import Dataset



class GeoDoubleAngleDataset(Dataset):

    def __init__(self, csv_data: pd.DataFrame, input_angle_cont_cols: List[List[str]],
                 input_angle_data_cols: List[List[str]], input_cont_cols: List[str],
                 input_cate_cols: List[str], task_target_cols: List[str]):
        self.csv_data = csv_data
        self.num_angle_feats = len(input_angle_cont_cols[0])
        self.num_angle_data = len(input_angle_data_cols[0])
        self.num_cont_feats = len(input_cont_cols)
        self.num_task_outs = len(task_target_cols)

        self.angle_feats_1 = csv_data[input_angle_cont_cols[0]].values.astype(np.float32)
        if len(input_angle_cont_cols) == 2:
            self.angle_feats_2 = csv_data[input_angle_cont_cols[1]].values.astype(np.float32)
        else:
            self.angle_feats_2 = None
        self.angle_data_1 = csv_data[input_angle_data_cols[0]].values.astype(np.float32)
        if len(input_angle_data_cols) == 2:
            self.angle_data_2 = csv_data[input_angle_data_cols[1]].values.astype(np.float32)
        else:
            self.angle_data_2 = None

        if input_cate_cols:
            self.num_cate_types = [len(set(csv_data[c])) for c in input_cate_cols]
            self.cate_feats_data = self.csv_data[input_cate_cols].values
        else:
            self.num_cate_types = None
            self.cate_feats_data = None

        self.cont_feats_data = self.csv_data[input_cont_cols].values.astype(np.float32)
        self.task_targets_data = self.csv_data[task_target_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        return {
            "ANGLE_FEAT_1": self.angle_feats_1[idx],
            "ANGLE_FEAT_2": self.angle_feats_2[idx] if self.angle_feats_2 is not None else 0,
            "ANGLE_DATA_1": self.angle_data_1[idx],
            "ANGLE_DATA_2": self.angle_data_2[idx] if self.angle_data_2 is not None else 0,
            "CONT_FEAT": self.cont_feats_data[idx],
            "CATE_FEAT": self.cate_feats_data[idx] if self.cate_feats_data is not None else 0,
            "TASK_TARGET": self.task_targets_data[idx],
        }
