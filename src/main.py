import warnings

warnings.filterwarnings("ignore")

from transformers import logging

logging.set_verbosity_error()

from parser import parse_args
from utils import set_seed
from data import load_data, CSATDataset
from models.model import load_model
from models.trainer import get_embeddings
from models.metrics import (
    eval_with_reg_ML,
    eval_with_cls_ML,
    print_reg_result,
    print_cls_result,
)

import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

from functools import partial


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


# Main flow
def main(args):

    # Seed settings
    set_seed(args.seed)

    # GPU settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data_path = f"../data/{args.dataset}"
    csat_kor_df = load_data(data_path=data_path)

    # Split data into train & test splits
    csat_train_df, csat_test_df = train_test_split(
        csat_kor_df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=csat_kor_df["difficulty"],
    )

    # Define CSAT korean dataset
    csat_train_dataset = CSATDataset(df=csat_train_df, task_type=args.task_type)
    csat_test_dataset = CSATDataset(df=csat_test_df, task_type=args.task_type)

    # Load Model & tokenizer
    if args.model_id == "qwen":
        # tokenizer, model = load_model("Qwen/Qwen3-Embedding-0.6B")
        # tokenizer, model = load_model("Qwen/Qwen3-Embedding-4B")
        # tokenizer, model = load_model("Qwen/Qwen3-Embedding-8B")
        pass
    elif args.model_id == "jin":
        # tokenizer, model = load_model("jinaai/jina-embeddings-v3")
        # tokenizer, model = load_model("jinaai/jina-embeddings-v4")
        pass
    elif args.model_id == "kbb":
        tokenizer, model = load_model("monologg/kobigbird-bert-base")
    elif args.model_id == "krl":
        tokenizer, model = load_model("vaiv/kobigbird-roberta-large")
    model = model.to(device)

    # Prepare dataloader
    train_dataloader = DataLoader(
        csat_train_dataset,
        args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, task_type=args.task_type),
    )
    test_dataloader = DataLoader(
        csat_test_dataset,
        args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, task_type=args.task_type),
    )

    # Define ML models
    reg_model_dict = {
        "Ridge": Ridge(random_state=args.seed),
        "SVR": SVR(verbose=0),
        "RandomForest": RandomForestRegressor(random_state=args.seed, verbose=0),
        "XGBoost": XGBRegressor(random_state=args.seed, verbose=0),
        "LightGBM": LGBMRegressor(random_state=args.seed, verbose=-1),
    }
    cls_model_dict = {
        "Ridge": RidgeClassifier(random_state=args.seed),
        "SVC": SVC(random_state=args.seed, verbose=0),
        "RandomForest": RandomForestClassifier(random_state=args.seed, verbose=0),
        "XGBoost": XGBClassifier(random_state=args.seed, verbose=0),
        "LightGBM": LGBMClassifier(random_state=args.seed, verbose=-1),
    }

    # Training or Inferencing directly
    metric_list = list()
    name_list = list()
    if args.train_flag == 0:
        X_train, y_train = get_embeddings(
            model, train_dataloader, device, pool_type=args.pool_type
        )
        X_test, y_test = get_embeddings(
            model, test_dataloader, device, pool_type=args.pool_type
        )

        # Evaluation
        train_set = (X_train, y_train)
        test_set = (X_test, y_test)
        if args.task_type == "reg":
            for name, model in reg_model_dict.items():
                metrics = eval_with_reg_ML(train_set, test_set, model)
                metric_list.append(metrics)
                name_list.append(name)
        else:
            for name, model in cls_model_dict.items():
                metrics = eval_with_cls_ML(train_set, test_set, model)
                metric_list.append(metrics)
                name_list.append(name)
    else:
        pass

    # Print result
    if args.task_type == "reg":
        print_reg_result(metric_list, name_list)
    else:
        print_cls_result(metric_list, name_list)


if __name__ == "__main__":
    args = parse_args()
    main(args)
