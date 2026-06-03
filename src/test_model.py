import config
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import f1_score, classification_report

print("1. Wczytywanie pełnego zbioru walidacyjnego do testu...")
df_test = pd.read_parquet(config.valdatapath)

df_test['text'] = df_test['language'] + " | " + df_test['code']
ds_test = Dataset.from_pandas(df_test[['text', 'label']])

print("2. Ładowanie przetrenowanego modelu z dysku...")
model_path = "./codebert_final_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256)

print("3. Tokenizacja zbioru walidacyjnego...")
ds_test = ds_test.map(tokenize_function, batched=True)
ds_test = ds_test.remove_columns(["text"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

test_args = TrainingArguments(
    output_dir="./test_wyniki",
    per_device_eval_batch_size=128,
    fp16=True,
    dataloader_num_workers=4,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=test_args,
    processing_class=tokenizer,
    data_collator=data_collator,
)

print("\nRozpoczynam wnioskowanie na GPU...")
predictions_output = trainer.predict(ds_test)

y_pred = np.argmax(predictions_output.predictions, axis=-1)
y_true = predictions_output.label_ids

print("\n========================================")
print(f"Ostateczne F1 na zbiorze walidacyjnym: {f1_score(y_true, y_pred, average='macro'):.4f}")
print("========================================")
print("\nSzczegółowy Raport:")
print(classification_report(y_true, y_pred))