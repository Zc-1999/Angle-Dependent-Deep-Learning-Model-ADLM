import math
from typing import List

import torch
import torch.nn as nn


class NumericalFeatureTokenizer(nn.Module):

    def __init__(self, n_feats: int, d_model: int):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(n_feats, d_model))
        self.bias = nn.Parameter(torch.Tensor(n_feats, d_model))

        nn.init.normal_(self.weight, std=1 / math.sqrt(d_model))
        nn.init.normal_(self.bias, std=1 / math.sqrt(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.weight.unsqueeze(0) * x.unsqueeze(-1)
        x = x + self.bias.unsqueeze(0)
        return x


class CategoricalFeatureTokenizer(nn.Module):

    def __init__(self, n_cates: List[int], d_model: int):
        super().__init__()

        category_offsets = torch.tensor([0] + n_cates[:-1]).cumsum(0)
        self.register_buffer("category_offsets", category_offsets, persistent=False)

        self.embedding = nn.Embedding(sum(n_cates), d_model)
        self.bias = nn.Parameter(torch.Tensor(len(n_cates), d_model))

        nn.init.normal_(self.bias, std=1 / math.sqrt(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x + self.category_offsets.unsqueeze(0))
        x + self.bias.unsqueeze(0)
        return x


class FeatureTokenizer(nn.Module):

    def __init__(self, n_feats: int, n_cates: List[int], d_model: int):
        super().__init__()

        self.numerical_tokenizer = NumericalFeatureTokenizer(n_feats, d_model)
        if n_cates:
            self.categorical_tokenizer = CategoricalFeatureTokenizer(n_cates, d_model)
        else:
            self.categorical_tokenizer = None

    def forward(self, x_cont: torch.Tensor, x_cate: torch.Tensor) -> torch.Tensor:
        if self.categorical_tokenizer:
            x = [self.numerical_tokenizer(x_cont), self.categorical_tokenizer(x_cate)]
            return torch.cat(x, dim=1)
        else:
            return self.numerical_tokenizer(x_cont)


class GlobalTokenizer(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(d_model))
        nn.init.normal_(self.weight, std=1 / math.sqrt(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return torch.cat([self.weight.expand(batch_size, 1, -1), x], dim=1)


class GeoTransformerEncoder(nn.Module):

    def __init__(self, n_feats: int, n_cates: List[int], d_model: int, n_head: int, n_layer: int, p_drop: float):
        super().__init__()

        self.feature_tokenizer = FeatureTokenizer(n_feats, n_cates, d_model)
        self.global_tokenizer = GlobalTokenizer(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=d_model,
                                                     nhead=n_head,
                                                     dim_feedforward=d_model * 2,
                                                     dropout=p_drop,
                                                     batch_first=True),
            num_layers=n_layer,
        )

    def forward(self, x_cont: torch.Tensor, x_cate: torch.Tensor) -> torch.Tensor:
        x = self.feature_tokenizer(x_cont, x_cate)
        x = self.global_tokenizer(x)
        x = self.transformer_encoder(x)
        return x[:, 0, :]


class MLPBlock(nn.Module):

    def __init__(self, d_model: int, p_drop: float):
        super().__init__()

        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p_drop)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.linear(x)))


class MLPDecoder(nn.Module):

    def __init__(self, d_model: int, n_outs: int, n_layer: int, p_drop: float):
        super().__init__()

        blocks = [MLPBlock(d_model, p_drop) for _ in range(n_layer)]
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(d_model, n_outs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return self.head(x)


class AngleDataTokenizer(nn.Module):

    def __init__(self, n_angle: int, d_model: int):
        super().__init__()
        self.linear = nn.Linear(n_angle * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_sin = torch.sin(x / 180 * math.pi)
        x_cos = torch.cos(x / 180 * math.pi)
        x = torch.cat([x_sin, x_cos], dim=1)
        return self.linear(x)


class GeoDoubleAngleEncoder(nn.Module):

    def __init__(self, n_angle_feats: int, n_angle_data: int, n_normal_feats: int, n_normal_cates: List[int],
                 d_model: int, n_head: int, n_layer: int, p_drop: float, double_angle: bool):
        super().__init__()

        self.angle_feature_tokenizer_1 = NumericalFeatureTokenizer(n_angle_feats, d_model)
        self.angle_data_tokenizer_1 = AngleDataTokenizer(n_angle_data, d_model)
        
        if double_angle:
            self.angle_feature_tokenizer_2 = NumericalFeatureTokenizer(n_angle_feats, d_model)
            self.angle_data_tokenizer_2 = AngleDataTokenizer(n_angle_data, d_model)
        else:
            self.angle_feature_tokenizer_2 = None
            self.angle_data_tokenizer_2 = None

        self.normal_tokenizer = FeatureTokenizer(n_normal_feats, n_normal_cates, d_model)
        self.global_tokenizer = GlobalTokenizer(d_model)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=d_model,
                                                     nhead=n_head,
                                                     dim_feedforward=d_model * 2,
                                                     dropout=p_drop,
                                                     batch_first=True),
            num_layers=n_layer,
        )

    def forward(self, x_angle_feats_1: torch.Tensor, x_angle_feats_2: torch.Tensor,
                x_angle_data_1: torch.Tensor, x_angle_data_2: torch.Tensor,
                x_cont: torch.Tensor, x_cate: torch.Tensor) -> torch.Tensor:
        x = self.normal_tokenizer(x_cont, x_cate)
        x_angle_feats_1 = self.angle_feature_tokenizer_1(x_angle_feats_1)
        x_angle_data_1 = self.angle_data_tokenizer_1(x_angle_data_1)
        x_1 = x_angle_feats_1 + x_angle_data_1.unsqueeze(1)

        if self.angle_feature_tokenizer_2 is not None:
            x_angle_feats_2 = self.angle_feature_tokenizer_2(x_angle_feats_2)
            x_angle_data_2 = self.angle_data_tokenizer_2(x_angle_data_2)
            x_2 = x_angle_feats_2 + x_angle_data_2.unsqueeze(1)
            x = torch.cat([x_1, x_2, x], dim=1)
        else:
            x = torch.cat([x_1, x], dim=1)
        
        x = self.global_tokenizer(x)
        x = self.transformer_encoder(x)
        return x[:, 0, :]
