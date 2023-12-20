import numpy as np
import faiss
from tqdm import trange
import csv
from transformers import BertTokenizer, BertModel

file = "D://apache(new).csv"
f = open(file, 'r', encoding='utf-8')
reader = list(csv.reader(f))[1:]
f.close()
log_list = list()
title_list = list()
resolution_list = list()
for i in trange(len(reader)):
    for k in reader[i][4:]:
        if len(k) > 0:
            log_list.append(k)
            title_list.append(reader[i][1][6:])
            resolution_list.append(reader[i][3])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

d = 768
index = faiss.IndexFlatL2(d)

for k in trange(len(log_list)):
    i = log_list[k]
    inputs = tokenizer(i, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)


    sentence1_vector = outputs.last_hidden_state[0, 0, :].detach().numpy()
    index.add(sentence1_vector.reshape(1, 768))


def get_simlog_solution(log, title, k):
    inputs = tokenizer(log, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    log_vec = outputs.last_hidden_state[0, 0, :].detach().numpy().reshape(1, 768)
    K = k
    D, I = index.search(log_vec, K)
    sim_logs = list()
    sim_resolutions = list()
    for j in I[0]:
        if title_list[j] != title:
            sim_logs.append(log_list[j])
            sim_resolutions.append(resolution_list[j])
    return sim_logs, sim_resolutions

