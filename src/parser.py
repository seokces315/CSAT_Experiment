import argparse


# Args parser
def parse_args():
    parser = argparse.ArgumentParser(description="CSAT_Project settings")

    parser.add_argument("--seed", default=42, type=int, help="Reproducibility")
    parser.add_argument("--dataset", default="CSAT_Kor.csv", type=str, help="Dataset")
    parser.add_argument("--test_size", default=0.2, type=int, help="Test size")
    parser.add_argument("--model_id", default="kbb", type=str, help="Model name")
    parser.add_argument("--train_flag", default=1, type=int, help="Train or not")
    parser.add_argument("--task_type", default="reg", type=str, help="Task type")
    parser.add_argument("--freeze_flag", default=1, type=int, help="Freeze or not")
    parser.add_argument("--fc_type", default="mlp", type=str, help="Network type")
    parser.add_argument("--pool_r", default=8, type=int, help="Pooling reduction rate")
    parser.add_argument("--pool_type", default="attn", type=str, help="Pooling type")
    parser.add_argument("--net_r", default=8, type=int, help="Network reduction rate")
    parser.add_argument("--m", default=2, type=int, help="CDAP's multiple rate")
    parser.add_argument("--alpha", default=0.5, type=float, help="Gating value")
    parser.add_argument(
        "--processor_type", default="gated", type=str, help="Processor type"
    )
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout rate")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    parser.add_argument("--cum_step", default=8, type=int, help="Accumulation steps")
    parser.add_argument("--epoch", default=10, type=int, help="Training epoch")
    parser.add_argument("--lr_rate", default=5e-5, type=float, help="Learning rate")
    parser.add_argument("--lr_type", default="linear", type=str, help="LR scheduler")
    parser.add_argument("--optim", default="adamw_torch", type=str, help="Optimizer")

    args = parser.parse_args()

    return args
