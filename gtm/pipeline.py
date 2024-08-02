import os
import pickle
from pprint import pprint
from typing import Dict

import lightgbm as lgb
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader

from gtm.dataset import GeoDoubleAngleDataset
from gtm.model import  LitGeoDoubleAngleModel



def run_doubleangle_pipeline(task_config: Dict):
    pprint(task_config)
    if not os.path.exists(task_config["output_folder_path"]):
        os.makedirs(task_config["output_folder_path"])

    train_data = pd.read_csv(task_config["train_data_path"])
    valid_data = pd.read_csv(task_config["valid_data_path"])

    if os.path.isdir(task_config["test_data_path"]):
        test_data_path = os.listdir(task_config["test_data_path"])
        test_data_path = [os.path.join(task_config["test_data_path"], p) for p in test_data_path]
    else:
        test_data_path = [task_config["test_data_path"]]

    standard_scaler = StandardScaler()
    train_data[task_config["dataset"]["input_cont_cols"]] = \
        standard_scaler.fit_transform(train_data[task_config["dataset"]["input_cont_cols"]])
    valid_data[task_config["dataset"]["input_cont_cols"]] = \
        standard_scaler.transform(valid_data[task_config["dataset"]["input_cont_cols"]])
    save_path = os.path.join(task_config["output_folder_path"], "input_cont_scaler.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(standard_scaler, f)

    if task_config["dataset"]["input_cate_cols"]:
        label_encoders = {}
        for c in task_config["dataset"]["input_cate_cols"]:
            label_encoder = LabelEncoder()
            train_data[c] = label_encoder.fit_transform(train_data[c])
            valid_data[c] = label_encoder.transform(valid_data[c])
            label_encoders[c] = label_encoder
        save_path = os.path.join(task_config["output_folder_path"], "input_cate_encoders.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(label_encoders, f)

    train_set = GeoDoubleAngleDataset(train_data, **task_config["dataset"])
    valid_set = GeoDoubleAngleDataset(valid_data, **task_config["dataset"])

    train_loader = DataLoader(train_set, shuffle=True, **task_config["dataloader"])
    valid_loader = DataLoader(valid_set, shuffle=False, **task_config["dataloader"])

    model = LitGeoDoubleAngleModel(
        n_angle_feats=train_set.num_angle_feats,
        n_angle_data=train_set.num_angle_data,
        n_normal_feats=train_set.num_cont_feats,
        n_normal_cates=train_set.num_cate_types,
        n_task_out=train_set.num_task_outs,
        **task_config["model"]
    )

    model_checkpoint = ModelCheckpoint(**task_config["callbacks"]["model_checkpoint"])
    early_stopping = EarlyStopping(**task_config["callbacks"]["early_stopping"])
    trainer = pl.Trainer(
        default_root_dir=task_config["output_folder_path"],
        callbacks=[model_checkpoint, early_stopping],
        **task_config["trainer"]
    )
    trainer.fit(model, train_loader, valid_loader)
    trainer.save_checkpoint(os.path.join(task_config["output_folder_path"], "double_angle_model.ckpt"))

    for i, data_path in enumerate(test_data_path):
        test_data = pd.read_csv(data_path)

        test_data[task_config["dataset"]["input_cont_cols"]] = \
            standard_scaler.transform(test_data[task_config["dataset"]["input_cont_cols"]])
        
        if task_config["dataset"]["input_cate_cols"]:
            for c in task_config["dataset"]["input_cate_cols"]:
                test_data[c] = label_encoders[c].transform(test_data[c])

        test_name = os.path.basename(test_data_path[i]).split(".")[0]
        test_set = GeoDoubleAngleDataset(test_data, **task_config["dataset"])
        test_loader = DataLoader(test_set, shuffle=False, **task_config["dataloader"])

        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            print("Best Checkpoint not Found! Using Current Weights for Prediction ...")
            ckpt_path = None
        predictions = trainer.predict(model, dataloaders=test_loader, ckpt_path=ckpt_path)
        predictions = torch.cat(predictions, dim=0).cpu().numpy()

        for j in range(len(task_config['dataset']['task_target_cols'])):
            task_name = task_config['dataset']['task_target_cols'][j]
            print(f"Task {task_name}:")
            y_true = test_set.task_targets_data[:, j]
            y_pred = predictions[:, j]
            print(f"RMSE: {mean_squared_error(y_true, y_pred) ** 0.5:.3f}")
            print(f"R2: {r2_score(y_true, y_pred):.3f}")
            test_data[f"{task_name}_PRED"] = y_pred
        test_data.to_csv(os.path.join(task_config["output_folder_path"], f"{test_name}_pred.csv"), index=False)


def run_doubleangle_inference_pipeline(task_config: Dict):
    pprint(task_config)
    if not os.path.exists(task_config["output_folder_path"]):
        os.makedirs(task_config["output_folder_path"])

    if os.path.isdir(task_config["test_data_path"]):
        test_data_path = os.listdir(task_config["test_data_path"])
        test_data_path = [os.path.join(task_config["test_data_path"], p) for p in test_data_path]
    else:
        test_data_path = [task_config["test_data_path"]]

    model = LitGeoDoubleAngleModel.load_from_checkpoint(task_config["model_checkpoint_path"])
    trainer = pl.Trainer(default_root_dir=task_config["output_folder_path"], **task_config["trainer"])

    for i, data_path in enumerate(test_data_path):
        test_data = pd.read_csv(data_path)

        with open(task_config["standard_scaler_path"], "rb") as f:
            standard_scaler = pickle.load(f)
        test_data[task_config["dataset"]["input_cont_cols"]] = \
            standard_scaler.transform(test_data[task_config["dataset"]["input_cont_cols"]])
        
        if task_config["dataset"]["input_cate_cols"]:
            with open(task_config["label_encoders_path"], "rb") as f:
                label_encoders = pickle.load(f)
            for c in task_config["dataset"]["input_cate_cols"]:
                test_data[c] = label_encoders[c].transform(test_data[c])

        test_name = os.path.basename(test_data_path[i]).split(".")[0]
        test_set = GeoDoubleAngleDataset(test_data, **task_config["dataset"])
        test_loader = DataLoader(test_set, shuffle=False, **task_config["dataloader"])

        predictions = trainer.predict(model, dataloaders=test_loader, ckpt_path=task_config["model_checkpoint_path"])
        predictions = torch.cat(predictions, dim=0).cpu().numpy()

        for j in range(len(task_config['dataset']['task_target_cols'])):
            task_name = task_config['dataset']['task_target_cols'][j]
            print(f"Task {task_name}:")
            y_true = test_set.task_targets_data[:, j]
            y_pred = predictions[:, j]
            print(f"RMSE: {mean_squared_error(y_true, y_pred) ** 0.5:.3f}")
            print(f"R2: {r2_score(y_true, y_pred):.3f}")
            test_data[f"{task_name}_PRED"] = y_pred
        test_data.to_csv(os.path.join(task_config["output_folder_path"], f"{test_name}_pred.csv"), index=False)
