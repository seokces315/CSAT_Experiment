import torch
import torch.nn as nn
import torch.nn.functional as F


# Mean Pooling Object
class MeanPooling(nn.Module):
    # Dummy initializer
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, hidden_state, attention_mask=None):
        if attention_mask is None:
            return hidden_state.mean(dim=1)

        # Prepare mask broadcasting
        mask = attention_mask.unsqueeze(-1).to(
            dtype=hidden_state.dtype, device=hidden_state.device
        )

        # Zero-out & Sum with given hidden states
        sum_embeddings = (hidden_state * mask).sum(dim=1)

        # Divide by number of non-masked(non-padded) tokens
        sum_mask = mask.sum(dim=1).clamp_min(torch.finfo(mask.dtype).tiny)
        embeddings = sum_embeddings / sum_mask

        return embeddings


# CLS Pooling Object
class CLSPooling(nn.Module):
    # Initializer
    def __init__(self, cls_idx):
        super(CLSPooling, self).__init__()
        self.cls_idx = cls_idx

    # Extract [CLS] token sequence representation from last hidden state
    def forward(self, hidden_state, attention_mask=None):
        return hidden_state[:, self.cls_idx, :]


# Attention Pooling Object
class AttentionPooling(nn.Module):
    # Initializer
    def __init__(self, fc_type, hidden_dim, r=8):
        super(AttentionPooling, self).__init__()
        bot_dim = hidden_dim // r
        if fc_type == "mlp":
            self.attn_fc = nn.Sequential(
                nn.Linear(hidden_dim, bot_dim),
                nn.Tanh(),
                nn.Linear(bot_dim, 1),
            )
        else:
            self.attn_fc = nn.Linear(hidden_dim, 1, bias=False)

    # Compute attention-weighted sequence representation
    def forward(self, hidden_state, attention_mask=None):
        # Get attention scores
        attn_scores = self.attn_fc(hidden_state)  # [B, L, 1]
        attn_scores = attn_scores.squeeze(-1).to(
            dtype=hidden_state.dtype, device=hidden_state.device
        )  # [B, L]

        # Masking
        if attention_mask is not None:
            attention_mask = attention_mask.bool()  # [B, L]
            neg_inf = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(~attention_mask, neg_inf)

        # Applying softmax
        attn_scores = attn_scores - attn_scores.max(dim=1, keepdim=True).values
        weights = F.softmax(attn_scores, dim=1)  # [B, L]

        # Weighted sum
        pooled = torch.einsum("bl,blh->bh", weights, hidden_state)

        return pooled


# Attention Pooling Object
class ChannelDependentAttentionPooling(nn.Module):
    # Initializer
    def __init__(self, fc_type, hidden_dim, r=8):
        super(ChannelDependentAttentionPooling, self).__init__()
        bot_dim = hidden_dim // r
        if fc_type == "mlp":
            self.cd_attn_fc = nn.Sequential(
                nn.Linear(hidden_dim, bot_dim),
                nn.Tanh(),
                nn.Linear(bot_dim, hidden_dim),
            )
        else:
            self.cd_attn_fc = nn.Linear(hidden_dim, hidden_dim, bias=False)

    # Compute channel-wise attention-weighted sequence representation
    def forward(self, hidden_state, attention_mask=None):
        # Get attention scores
        cd_attn_scores = self.cd_attn_fc(hidden_state)  # [B, L, H]
        cd_attn_scores = cd_attn_scores.to(
            dtype=hidden_state.dtype, device=hidden_state.device
        )

        # Masking
        if attention_mask is not None:
            attention_mask = attention_mask.bool()  # [B, L]
            neg_inf = torch.finfo(cd_attn_scores.dtype).min
            cd_attn_scores = cd_attn_scores.masked_fill(
                ~attention_mask.unsqueeze(-1), neg_inf
            )

        # Applying softmax
        cd_attn_scores = cd_attn_scores - cd_attn_scores.max(dim=1, keepdim=True).values
        cd_weights = F.softmax(cd_attn_scores, dim=1)  # [B, L, H]

        # Weighted sum
        pooled = torch.einsum("blh,blh->bh", cd_weights, hidden_state)

        return pooled


# Function for mean pooling
def mean_pooling(output, attention_mask=None):
    # Get last hidden state layer
    last_hidden_state = output.last_hidden_state

    if attention_mask is None:
        return last_hidden_state.mean(dim=1)

    # Prepare mask broadcasting
    mask = attention_mask.unsqueeze(-1).to(
        dtype=last_hidden_state.dtype, device=last_hidden_state.device
    )

    # Zero-out & Sum with given hidden states
    sum_embeddings = (last_hidden_state * mask).sum(dim=1)

    # Divide by number of non-masked(non-padded) tokens
    sum_mask = mask.sum(dim=1).clamp_min(torch.finfo(mask.dtype).tiny)
    embeddings = sum_embeddings / sum_mask

    return embeddings
