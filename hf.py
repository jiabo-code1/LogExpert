


from datasets import load_dataset_builder
from datasets import load_dataset
from transformers import AutoTokenizer
import csv
import torch
from tqdm import trange
from transformers import AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

dataset = load_dataset("../log_recognizer_dataset")['train']


def tokenization(example):
    return tokenizer(example['text'], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenization, batched=True)
tokenized_datasets = tokenized_datasets.train_test_split(seed=42)
train_dataset = tokenized_datasets['train']
eval_dataset = tokenized_datasets['test']

model = AutoModelForSequenceClassification.from_pretrained("./test_trainer3//checkpoint-500")
from transformers import TrainingArguments

import numpy as np
import evaluate

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load('f1')
metric = evaluate.combine([accuracy, precision, recall, f1])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


from transformers import TrainingArguments, Trainer


training_args = TrainingArguments(output_dir="test_trainer4", evaluation_strategy="epoch", push_to_hub=True)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)



trainer.train()



def eva():
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load('f1')
    metric = evaluate.combine([accuracy, precision, recall, f1])
    references = eval_dataset['label']
    model = AutoModelForSequenceClassification.from_pretrained("./test_trainer3/checkpoint-500")
    for i in trange(len(eval_dataset)):
        tokenized_input = tokenizer(eval_dataset[i]['text'], return_tensors='pt', padding=True, truncation=True)
        prediction = np.argmax(model(**tokenized_input).logits.detach().numpy(),axis=-1)
        reference = eval_dataset[i]['label']
        metric.add_batch(references=[reference], predictions=[prediction])
    result = metric.compute()
    print(result)


