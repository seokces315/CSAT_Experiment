import argparse


# Args parser
def parse_args():
    parser = argparse.ArgumentParser(description="CSAT_Project settings")

    parser.add_argument("--seed", default=42, type=int, help="Reproducibility")
    parser.add_argument("--dataset", default="CSAT_Kor.csv", type=str, help="Dataset")
    parser.add_argument("--test_size", default=0.2, type=int, help="Test size")
    parser.add_argument("--model_id", default="kbb", type=str, help="Model name")
    parser.add_argument("--task_type", default="reg", type=str, help="Task type")
    parser.add_argument("--pool_type", default="cls", type=str, help="Pooling type")
    parser.add_argument("--train_flag", default=0, type=int, help="Train or not")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")

    args = parser.parse_args()

    return args
