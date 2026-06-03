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
from sklearn.metrics import f1_score, accuracy_score, classification_report

print("Wczytywanie danych...")
df_train = pd.read_parquet(config.traindatapath)
df_val = pd.read_parquet(config.valdatapath)

df_train = df_train.sample(50000, random_state=42)
df_val = df_val.sample(min(10000, len(df_val)), random_state=42)

df_train['text'] = df_train['language'] + " | " + df_train['code']
df_val['text'] = df_val['language'] + " | " + df_val['code']

ds_train = Dataset.from_pandas(df_train[['text', 'label']])
ds_val = Dataset.from_pandas(df_val[['text', 'label']])

model_name = "microsoft/codebert-base"
print(f"Pobieranie tokenizera i modelu {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256)

print("Tokenizacja zbiorów...")
ds_train = ds_train.map(tokenize_function, batched=True)
ds_val = ds_val.map(tokenize_function, batched=True)

ds_train = ds_train.remove_columns(["text"])
ds_val = ds_val.remove_columns(["text"])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    macro_f1 = f1_score(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1_macro": macro_f1}

training_args = TrainingArguments(
    output_dir="./codebert_wyniki",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    report_to="none"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\nRozpoczynam Pełny Fine-Tuning na GPU...")
trainer.train()

print("\nGenerowanie ostatecznego raportu z najlepszego modelu...")
predictions_output = trainer.predict(ds_val)
y_pred = np.argmax(predictions_output.predictions, axis=-1)
y_true = predictions_output.label_ids

print(f"Końcowe F1 (macro): {f1_score(y_true, y_pred, average='macro'):.4f}")
print("\nSzczegółowy Raport:")
print(classification_report(y_true, y_pred))

trainer.save_model("./codebert_finalny_model")
print("Model zapisany poprawnie w folderze './codebert_finalny_model'!")