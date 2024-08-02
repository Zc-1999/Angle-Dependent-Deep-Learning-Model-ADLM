from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import R2Score, PearsonCorrCoef

from gtm.loss import RelativeErrorLoss
from gtm.module import (
    GeoDoubleAngleEncoder,
    MLPDecoder
)


class LitGeoDoubleAngleModel(pl.LightningModule):

    def __init__(self, n_angle_feats: int, n_angle_data: int, n_normal_feats: int, n_normal_cates: List[int],
                 n_task_out: int, d_model: int, n_tf_head: int, n_tf_layer: int, p_tf_drop: float,
                 n_mlp_layer: int, p_mlp_drop: float, lr: float, double_angle: bool = True, loss: str = "mse"):
        super().__init__()
        self.save_hyperparameters()

        self.geo_encoder = GeoDoubleAngleEncoder(n_angle_feats, n_angle_data, n_normal_feats, n_normal_cates,
                                                 d_model, n_tf_head, n_tf_layer, p_tf_drop, double_angle)
        self.task_decoder = MLPDecoder(d_model, n_task_out, n_mlp_layer, p_mlp_drop)

        self.lr = lr
        if loss == "mse":
            self.criterion = nn.MSELoss()
        elif loss == "rel":
            self.criterion = RelativeErrorLoss(a=0.15, b=0.05)
        else:
            raise ValueError("Loss is not Supported!")
        self.metric_r2 = R2Score(num_outputs=n_task_out)

    def forward(self, x_angle_feats_1, x_angle_feats_2, x_angle_data_1, x_angle_data_2, x_cont, x_cate):
        x = self.geo_encoder(x_angle_feats_1, x_angle_feats_2, x_angle_data_1, x_angle_data_2, x_cont, x_cate)
        return self.task_decoder(x)

    def _shared_step(self, batch):
        return self(batch["ANGLE_FEAT_1"], batch["ANGLE_FEAT_2"],
                    batch["ANGLE_DATA_1"], batch["ANGLE_DATA_2"],
                    batch["CONT_FEAT"], batch["CATE_FEAT"])

    def training_step(self, batch, batch_idx):
        outputs = self._shared_step(batch)
        loss = self.criterion(outputs, batch["TASK_TARGET"])
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self._shared_step(batch)
        loss = self.criterion(outputs, batch["TASK_TARGET"])

        self.metric_r2(outputs, batch["TASK_TARGET"])
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_r2", self.metric_r2, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        return self._shared_step(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-3)
        return optimizer
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer=optimizer,
        #     T_0=2,
        #     T_mult=2,
        #     eta_min=1e-5
        # )
        # return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
