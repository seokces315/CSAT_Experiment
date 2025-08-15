from .pooling import mean_pooling

from tqdm import tqdm

import torch
import torch.nn as nn

import numpy as np


# MSE + MAE Type Loss
class HuberLoss(nn.Module):
    # Initializer
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, preds, targets):
        assert preds.shape == targets.shape

        # Define the values of abs_error, delta
        delta = torch.tensor(self.delta, dtype=preds.dtype, device=preds.device)
        abs_error = (preds - targets).abs()

        # Flag
        is_small_error = abs_error <= delta

        # Compute & Select loss
        small_error_loss = 0.5 * (abs_error**2)
        large_error_loss = delta * (abs_error - 0.5 * delta)
        loss = torch.where(is_small_error, small_error_loss, large_error_loss)

        return loss.mean()


# Function for custom collate with dynamic padding
def collate_fn(batch, tokenizer, task_type):
    # None sample filtering
    batch = [item for item in batch if item is not None]
    texts = [item["text"] for item in batch]
    labels = [item["label"] for item in batch]

    encoded_texts = tokenizer(
        texts,
        padding="longest",
        truncation=True,
        max_length=2048,
        return_tensors="pt",
    )

    labels = (
        torch.tensor(labels, dtype=torch.float)
        if task_type == "reg"
        else torch.tensor(labels, dtype=torch.long)
    )

    return {
        "input_ids": encoded_texts["input_ids"],
        "attention_mask": encoded_texts["attention_mask"],
        "labels": labels,
    }


# Wrapping function for HuggingFace Trainer
def wrap_collate_fn(tokenizer, task_type):
    def collate_fn_(batch):
        # None sample filtering
        batch = [item for item in batch if item is not None]
        texts = [item["texts"] for item in batch]
        labels = [item["labels"] for item in batch]

        encoded_texts = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )

        labels = (
            torch.tensor(labels, dtype=torch.float)
            if task_type == "reg"
            else torch.tensor(labels, dtype=torch.long)
        )

        return {
            "input_ids": encoded_texts["input_ids"],
            "attention_mask": encoded_texts["attention_mask"],
            "labels": labels,
        }

    return collate_fn_


# Function to get embeddings from pre-trained model
def get_embeddings(embedding_model, dataloader, device, pool_type):
    # Local vars
    embedding_list = list()
    label_list = list()

    # Evaluation mode
    embedding_model.eval()

    # Embedding loop
    for batch in tqdm(dataloader):
        # BatchEncoding Data -> GPU
        labels = batch.pop("labels").numpy()
        input_dicts = {k: v.to(device) for k, v in batch.items()}

        # Inferencing
        with torch.no_grad():
            output = embedding_model(**input_dicts)

            # 1. Mean pooling
            if pool_type == "mean":
                attention_mask = input_dicts["attention_mask"]
                mean_pooled = mean_pooling(output, attention_mask)
                embeddings = mean_pooled.float().cpu().numpy()
            # 2. CLS Token
            else:
                embeddings = output.pooler_output.float().cpu().numpy()

            # Append to list, respectively
            embedding_list.extend(embeddings)
            label_list.extend(labels)

    return np.array(embedding_list), np.array(label_list)
