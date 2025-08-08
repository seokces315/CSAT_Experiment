from .pooling import mean_pooling

from tqdm import tqdm

import torch

import numpy as np


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
