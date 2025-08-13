from transformers import AutoModel, AutoTokenizer

from .pooling import (
    MeanPooling,
    CLSPooling,
    AttentionPooling,
    ChannelDependentAttentionPooling,
)
from .trainer import HuberLoss

import torch
import torch.nn as nn


# Linear Head Layer Class
class LinearHead(nn.Module):
    # Initializer
    def __init__(self, hidden_size, out_dim, dropout=None, r=None):
        super(LinearHead, self).__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear_layer = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        x = self.norm(x)
        return self.linear_layer(x)


# Linear Head Layer Class + Dropout
class LinearHead_Rglr(nn.Module):
    # Initializer
    def __init__(self, hidden_size, out_dim, dropout, r=None):
        super(LinearHead_Rglr, self).__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear_layer = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.dropout(x)
        return self.linear_layer(x)


# MLP Head Layer Class
class MLPHead(nn.Module):
    # Initializer
    def __init__(self, hidden_size, out_dim, dropout, r=8):
        super(MLPHead, self).__init__()
        bot_dim = hidden_size // r
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, bot_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bot_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# Gated-MLP Head Layer Class
class GatedGELUHead(nn.Module):
    # Initializer
    def __init__(self, hidden_size, out_dim, dropout, m=2, alpha=0.5):
        super(GatedGELUHead, self).__init__()
        gated_dim = hidden_size * m
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.norm = nn.LayerNorm(hidden_size)
        self.fc_g = nn.Linear(hidden_size, gated_dim, bias=False)  # Gate
        self.fc_h = nn.Linear(hidden_size, gated_dim, bias=False)  # Value
        self.gelu = nn.GELU()
        self.proj = nn.Linear(gated_dim, hidden_size, bias=False)  # Projection Layer
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, out_dim)  # Output Layer

    def forward(self, x):
        # Gated path, normalization
        x_norm = self.norm(x)

        # Compute gated vector, value vector
        g = self.gelu(self.fc_g(x_norm))
        h = self.fc_h(x_norm)

        # Gating & Projection
        delta = self.dropout(self.proj(g * h))

        # Residual connection
        x = x + self.alpha * delta

        return self.out(x)


# Embedding Model + Regressor/Classifier
class KoBigBirdProcessor(nn.Module):
    # Initializer
    def __init__(
        self,
        task_type,
        embedding_model,
        fFlag,
        fc_type,
        pool_r,
        pooling_type,
        net_r,
        m,
        alpha,
        processor_type,
        dropout,
        num_targets,
    ):
        super(KoBigBirdProcessor, self).__init__()

        self.task_type = task_type
        self.embedding_model = embedding_model
        self.hidden_dim = self.embedding_model.config.hidden_size
        self.fFlag = fFlag

        # Freeze or Not
        if self.fFlag is True:
            for param in self.embedding_model.parameters():
                param.requires_grad = False
            self.embedding_model.eval()

        # Define pooling map
        pooling_map = {
            "mean": MeanPooling(),
            "cls": CLSPooling(cls_idx=0),
            "attn": AttentionPooling(
                fc_type=fc_type, hidden_dim=self.hidden_dim, r=pool_r
            ),
            "cd_attn": ChannelDependentAttentionPooling(
                fc_type=fc_type, hidden_dim=self.hidden_dim, r=pool_r
            ),
        }
        self.pool = pooling_map[pooling_type]

        # Define processor map
        processor_map = {
            "ln": LinearHead(
                self.hidden_dim,
                self.num_targets,
                dropout=dropout,
                r=net_r,
            ),
            "ln_rglr": LinearHead_Rglr(
                self.hidden_dim,
                self.num_targets,
                dropout=dropout,
                r=net_r,
            ),
            "mlp": MLPHead(
                self.hidden_dim,
                self.num_targets,
                dropout=dropout,
                r=net_r,
            ),
            "gated": GatedGELUHead(
                self.hidden_dim,
                self.num_targets,
                dropout=dropout,
                m=m,
                alpha=alpha,
            ),
        }
        self.processor = processor_map[processor_type]

        self.num_targets = num_targets

    def forward(self, **input_dicts):
        # Backbone inferencing
        if self.fFlag is True:
            with torch.no_grad():
                outputs = self.embedding_model(**input_dicts)
        else:
            outputs = self.embedding_model(**input_dicts)

        # Pooling
        attention_mask = input_dicts["attention_mask"]
        pooled = self.pool(outputs.last_hidden_state, attention_mask)

        # Processing Head Layer
        logits = (
            self.processor(pooled).squeeze(-1)
            if self.task_type == "reg"
            else self.processor(pooled)
        )

        # Calculate loss
        labels = input_dicts["labels"]
        if self.task_type == "reg":
            labels = labels.float().to(device=logits.device).squeeze(-1)
            criterion = nn.L1Loss()
            # criterion = HuberLoss()
        else:
            labels = labels.long().to(device=logits.device).squeeze(-1)
            criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)

        return {"logits": logits, "loss": loss}


# Function to load tokenizer & model
def load_model(model_id):
    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

    return tokenizer, model
