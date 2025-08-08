import os
import sys

# Get parent folder path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Append to sys.path
sys.path.append(parent_dir)

from torch.utils.data import Dataset

import pandas as pd

import ast


# Custom dataset for CSAT-Kor-Project
class CSATDataset(Dataset):
    def __init__(self, df, task_type):
        self.df = df
        self.task_type = task_type
        self.counter = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            choices = ast.literal_eval(row["choices"])

            prompt = (
                f"[시험 종류]\n{row['exam_type']}\n\n"
                f"[지문]\n{row['paragraph']}\n\n"
                f"[문제 유형]\n{row['question_type']}\n\n"
                f"[문제]\n{row['question']}\n\n"
                f"[보기]\n{row['question_plus']}\n\n"
                f"[선택지]\n1. {choices[0]}\n2. {choices[1]}\n3. {choices[2]}\n4. {choices[3]}\n5. {choices[4]}\n\n"
                f"[정답]\n{row['answer']}"
            )

            text = prompt.replace("[보기]\nnan\n\n", "")

            if self.task_type == "reg":
                label = row["answer_rate"]
            else:
                label = row["difficulty"]

            return {"text": text, "label": label}

        except Exception as e:
            # print(row["paragraph"])
            self.counter += 1
            return None


# Function to classify exam's style
def classify_exam(splits):
    month = int(splits[3])

    if month in [3, 4, 5, 7, 10]:
        exam = "학력평가"
    elif month in [6, 9]:
        exam = "모의고사"
    else:
        exam = "수능"

    return exam


# Function to label answer rate
def rate2label(answer_rate):
    return 2 if answer_rate < 0.5 else 1 if answer_rate < 0.8 else 0


# Function to re-define given data
def load_data(data_path):
    # Load csv file
    csat_kor_df = pd.read_csv(data_path)

    # Data transformation
    csat_kor_df["question_id_split"] = csat_kor_df["question_id"].map(
        lambda x: x.split("_")[1:] if x.split("_")[0] == "Odd" else x.split("_")
    )
    csat_kor_df["exam_type"] = csat_kor_df["question_id_split"].map(classify_exam)
    csat_kor_df["paragraph"] = (
        csat_kor_df["paragraph"]
        + (
            " " + csat_kor_df["paragraph_image_description"].fillna("").astype(str)
        ).str.strip()
    )
    csat_kor_df["question_plus"] = (
        csat_kor_df["image_description"].fillna("").astype(str) + " "
    ).str.strip() + csat_kor_df["question_plus"]
    csat_kor_df["answer"] = csat_kor_df["answer"].astype(int)
    csat_kor_df["answer_rate"] = csat_kor_df["answer_rate"].map(
        lambda x: round(0.01 * x, 2)
    )
    csat_kor_df["difficulty"] = csat_kor_df["answer_rate"].map(rate2label)

    # Reorder DataFrame's columns
    new_columns = [
        "exam_type",
        "paragraph",
        "question_type",
        "question",
        "question_plus",
        "choices",
        "answer",
        "answer_rate",
        "difficulty",
    ]
    csat_kor_df = csat_kor_df[new_columns]

    return csat_kor_df


if __name__ == "__main__":
    # Data related vars
    dataset_name = "CSAT_Kor.csv"
    data_path = f"{parent_dir}/data/{dataset_name}"
    task_type = "reg"  # or "cls"

    # Define CSAT Kor dataset
    csat_kor_df = load_data(data_path)
    csat_kor_dataset = CSATDataset(df=csat_kor_df, task_type=task_type)
    print(csat_kor_dataset[0]["label"])
