"""Бүх туршилтын нэгдсэн хуваалт.

train : 2015-01 .. 2021-12
valid : 2022 (early stopping)
test  : 2023-01 .. 2025-04-07 (дипломын үндсэн үр дүн; анхны календарь энд дуусдаг)
forward: 2025-04-08 .. (календарийг сунгасны дараах, загварыг огт өөрчлөөгүй "жинхэнэ ирээдүй" шалгалт —
         scripts/eval_forward.py)
"""
import pandas as pd

TRAIN_END = pd.Timestamp("2022-01-01", tz="UTC")
VALID_END = pd.Timestamp("2023-01-01", tz="UTC")
TEST_END = pd.Timestamp("2025-04-08", tz="UTC")
