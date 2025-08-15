import warnings

warnings.filterwarnings("ignore")

from transformers import logging

logging.set_verbosity_error()

import wandb

wandb.init(project="CSAT_Experiment")

from parser import parse_args
from utils import set_seed, is_bf16_supported
from data import load_data, CSATDataset
from models.model import EmbeddingProcessor, load_model
from models.trainer import collate_fn, wrap_collate_fn, get_embeddings
from models.metrics import (
    reg_metrics,
    cls_metrics,
    eval_with_reg_ML,
    eval_with_cls_ML,
    print_reg_result,
    print_cls_result,
)

import os
import json

import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from functools import partial


# Main flow
def main(args):

    # Seed settings
    set_seed(args.seed)

    # GPU settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data_path = f"../data/{args.dataset}"
    csat_kor_df = load_data(data_path=data_path)

    # Load Model & tokenizer
    bf16_flag = is_bf16_supported()
    tokenizer, model = None, None
    if args.model_id == "qwen":
        # tokenizer, model = load_model("Qwen/Qwen3-Embedding-0.6B", bf16_flag=bf16_flag)
        # tokenizer, model = load_model("Qwen/Qwen3-Embedding-4B", bf16_flag=bf16_flag)
        # tokenizer, model = load_model("Qwen/Qwen3-Embedding-8B", bf16_flag=bf16_flag)
        pass
    elif args.model_id == "jin":
        # tokenizer, model = load_model("jinaai/jina-embeddings-v3", bf16_flag=bf16_flag)
        # tokenizer, model = load_model("jinaai/jina-embeddings-v4", bf16_flag=bf16_flag)
        pass
    elif args.model_id == "kbb":
        tokenizer, model = load_model(
            "monologg/kobigbird-bert-base", bf16_flag=bf16_flag
        )
    elif args.model_id == "krl":
        tokenizer, model = load_model(
            "vaiv/kobigbird-roberta-large", bf16_flag=bf16_flag
        )

    # Training or Inferencing directly
    metric_list = list()
    name_list = list()
    if args.train_flag == 0:
        model = model.to(device)

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

        # Prepare dataloader
        train_dataloader = DataLoader(
            csat_train_dataset,
            args.batch_size,
            shuffle=True,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                task_type=args.task_type,
            ),
        )
        test_dataloader = DataLoader(
            csat_test_dataset,
            args.batch_size,
            shuffle=False,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                task_type=args.task_type,
            ),
        )

        # Get embeddings to be used in ML training/testing
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

        # Print results
        if args.task_type == "reg":
            print_reg_result(metric_list, name_list)
        else:
            print_cls_result(metric_list, name_list)

    else:
        num_targets = 1 if args.task_type == "reg" else 3
        task_model = EmbeddingProcessor(
            model,
            args.task_type,
            args.freeze_flag,
            args.fc_type,
            args.pool_r,
            args.pool_type,
            args.net_r,
            args.m,
            args.alpha,
            args.processor_type,
            args.dropout,
            num_targets,
        )
        task_model = task_model.to(device)

        # Split data into train & test splits
        csat_train_df, csat_test_df = train_test_split(
            csat_kor_df,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=csat_kor_df["difficulty"],
        )

        csat_eval_df, csat_test_df = train_test_split(
            csat_test_df,
            test_size=0.5,
            random_state=args.seed,
            stratify=csat_test_df["difficulty"],
        )

        # Define CSAT korean dataset
        csat_train_dataset = CSATDataset(df=csat_train_df, task_type=args.task_type)
        csat_eval_dataset = CSATDataset(df=csat_eval_df, task_type=args.task_type)
        csat_test_dataset = CSATDataset(df=csat_test_df, task_type=args.task_type)

        # Define training arguments
        greater_is_better = False if args.task_type == "reg" else True
        metric_for_best_model = (
            "eval_mae" if args.task_type == "reg" else "eval_accuracy"
        )
        training_args = TrainingArguments(
            output_dir="./output",
            data_seed=args.seed,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.cum_step,
            num_train_epochs=args.epoch,
            learning_rate=args.lr_rate,
            lr_scheduler_type=args.lr_type,
            optim=args.optim,
            logging_strategy="steps",
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=1,
            remove_unused_columns=False,
            dataloader_drop_last=True,
            disable_tqdm=False,
            full_determinism=True,
            load_best_model_at_end=True,
            greater_is_better=greater_is_better,
            metric_for_best_model=metric_for_best_model,
            run_name="CSAT_Experiment",
            report_to="wandb",
        )

        # Define trainer for training
        data_collator = wrap_collate_fn(tokenizer=tokenizer, task_type=args.task_type)
        compute_metrics = reg_metrics if args.task_type == "reg" else cls_metrics
        trainer = Trainer(
            model=task_model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=csat_train_dataset,
            eval_dataset=csat_eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=5, early_stopping_threshold=1e-3
                )
            ],
        )

        # Train process
        trainer.train()

        # Test process
        test_metrics = trainer.evaluate(eval_dataset=csat_test_dataset)

        # Save results
        save_path = os.path.join(
            "./res",
            f"{args.model_id}_{args.task_type}_{args.pool_type}_{args.processor_type}_test_metrics.json",
        )

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
