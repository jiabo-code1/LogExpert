import csv

import pandas as pd
import openai
import os
import gpt_turbo
from tqdm import trange
from evaluate_metric import get_BLEU, get_ROUGE
file = "D://AIOps-Event-Graph-WebData-main//apache(new).csv"
new_file = "D://AIOps-Event-Graph-WebData-main//response(tem=0.7 gpt4).csv"
f = open(file, 'r', encoding='utf-8')
f2 = open(new_file, 'a', encoding='utf-8')
writer = csv.writer(f2)
reader = list(csv.reader(f))[1:]
bleu1_sum = 0
bleu2_sum = 0
bleu3_sum = 0
rouge1_sum = 0
rouge2_sum = 0
rouge3_sum = 0
for k in trange(0, len(reader)): # Apache WSGI permission error using django log
    i = reader[k]                        #AWS elastic beanstalk 100.0 % of the r
    row_data = list()
    title = i[1][6:]
    description = i[2][12:]


    # resolution = i[3][11:]
    resolution = i[3]
    # resolution=resolution.replace('\n','')

    log = i[4]


    answer1 = gpt_turbo.get_response_lognoly_gpt4(log)


    #answer2 = gpt_turbo.get_response_log_description(log, description)
    answer2='test'

    answer3 = gpt_turbo.get_response_proposed_gpt4(log, title)
    row_data.append(title)
    row_data.append(description)
    row_data.append(resolution)
    row_data.append(log)
    row_data.append(answer1)
    row_data.append(answer2)
    row_data.append(answer3)
    writer.writerow(row_data)
'''
    bleu1 = get_BLEU(resolution, answer1)
    bleu2 = get_BLEU(resolution, answer2)
    bleu3 = get_BLEU(resolution, answer3)
    rouge1 = get_ROUGE(resolution, answer1)
    rouge2 = get_ROUGE(resolution, answer2)
    rouge3 = get_ROUGE(resolution, answer3)
    print(
        f"bleu1{bleu1},bleu2{bleu2},bleu3{bleu3},rouge1{rouge1},rouge2{rouge2}.rouge3{rouge3}")

    bleu1_sum = bleu1_sum + bleu1
    bleu2_sum = bleu2_sum + bleu2
    bleu3_sum = bleu3_sum + bleu3
    rouge1_sum = rouge1_sum + rouge1
    rouge2_sum = rouge2_sum + rouge2
    rouge3_sum = rouge3_sum + rouge3
# print(f"bleu1{bleu1_sum},bleu2{bleu2_sum},bleu3{bleu3_sum},rouge1{rouge1_sum},rouge2{rouge2_sum}.rouge3{rouge3_sum}")
'''
f.close()
f2.close()


def ev():
    import csv
    from evaluate_metric import get_BLEU, get_ROUGE, get_bert_score
    file = "response(tem=0.5).csv"
    f = open(file, 'r', encoding='utf-8')
    reader = list(csv.reader(f))
    file2 = "score(tem=0.5).csv"
    f2 = open(file2, 'w', encoding='utf-8')
    writer = csv.writer(f2)
    bleu1_sum = 0
    bleu2_sum = 0
    bleu3_sum = 0

    rouge1_sum = 0
    rouge2_sum = 0
    rouge3_sum = 0

    bert_score1_sum = 0
    bert_score2_sum = 0
    bert_score3_sum = 0
    num = 0
    for i in reader:
        print(i)
        num = num + 1
        title = i[1]
        answer1 = i[4]
        answer2 = i[5]
        answer3 = i[6]
        resolution = i[3]
        if num < 5:
            print(f'计算得分的两段文本：{resolution}\n,第二段：{answer3}')
        bleu1 = get_BLEU(resolution, answer1)
        bleu2 = get_BLEU(resolution, answer2)
        bleu3 = get_BLEU(resolution, answer3)
        rouge1 = get_ROUGE(resolution, answer1)
        rouge2 = get_ROUGE(resolution, answer2)
        rouge3 = get_ROUGE(resolution, answer3)
        bert_score1 = get_bert_score(resolution, answer1)
        bert_score2 = get_bert_score(resolution, answer2)
        bert_score3 = get_bert_score(resolution, answer3)
        n1 = round(bleu1, 2)
        n2 = round(bleu2, 2)
        n3 = round(bleu3, 2)
        n4 = round(rouge1, 2)
        n5 = round(rouge2, 2)
        n6 = round(rouge3, 2)
        n7 = round(float(bert_score1), 2)
        n8 = round(float(bert_score2), 2)
        n9 = round(float(bert_score3), 2)
        ns = [resolution, answer3, n1, n2, n3, n4, n5, n6, n7, n8, n9]
        ns.append(title)

        print(n1, n2, n3, n4, n5, n6, n7, n8, n9)
        writer.writerow(ns)
        # print(
        #    f"bleu1{bleu1},bleu2{bleu2},bleu3{bleu3},rouge1{rouge1},rouge2{rouge2}.rouge3{rouge3}")

        bleu1_sum = bleu1_sum + bleu1
        bleu2_sum = bleu2_sum + bleu2
        bleu3_sum = bleu3_sum + bleu3
        rouge1_sum = rouge1_sum + rouge1
        rouge2_sum = rouge2_sum + rouge2
        rouge3_sum = rouge3_sum + rouge3
        bert_score1_sum = bert_score1_sum + bert_score1
        bert_score2_sum = bert_score2_sum + bert_score2
        bert_score3_sum = bert_score3_sum + bert_score3
    print(bleu1_sum, bleu2_sum, bleu3_sum, rouge1_sum, rouge2_sum, rouge3_sum, bert_score1_sum, bert_score2_sum,
          bert_score3_sum)
    f.close()
    f2.close()



def resolution_correct(file):
    file = "D://response(tem=0.5 gpt4).csv"
    file2 = "D://apache.csv"
    df = pd.read_csv(file)
    df2 = pd.read_csv(file2)

    df.columns = ['title', 'description', 'resolution', 'log', 'answer1', 'answer2', 'answer3']
    df = df.drop('log', axis=1)
    for i in range(len(df)):
        title = df.iloc[i]['title']
        for k in range(len(df2)):
            if df2.iloc[k]['title'][6:] == title:
                df.replace(df.iloc[i]['title'], df2.iloc[k]['title'], inplace=True)
                df.replace(df.iloc[i]['resolution'], df2.iloc[k]['resolution'], inplace=True)
    df.to_csv(file)


