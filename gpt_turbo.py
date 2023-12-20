import csv
import json

import openai
import os
import requests
import find_simlog
from evaluate_metric import get_BLEU, get_ROUGE
from transformers import AutoTokenizer




def askChatGPT_logonly(log):
    MODEL = "gpt-3.5-turbo-16k"
    messages = [{"role": "system", "content": "You are an helpful assistant."}, {"role": "user",
    "content": f'''My system output an error log. You should give the executable steps to fix the error log.
     The error log:{log}'''}, ]

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0.5)
    return response['choices'][0]['message']['content']



def askChatGPT_logonly_gpt4(log):
    MODEL = "gpt-4"
    messages = [{"role": "system", "content": "You are an helpful assistant."}, {"role": "user",
    "content": f'''My system output an error log. You should give the executable steps to fix the error log.
     The error log:{log}'''}, ]

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0.5)
    return response['choices'][0]['message']['content']



def askChatGPT_log_description(log, description):
    MODEL = "gpt-3.5-turbo-16k"
    messages = [{"role": "system", "content": "You are an helpful assistant."}, {"role": "user",
    "content": f'''I have encountered an error. 
    I will provide you with [the main error log] and [the description of the error]. 
    The description of the error is :{description}
    Notification: Combine your knowledge about the main error log with the key knowledge mentioned in the description. 
    You should only output the steps which are executable to fix [the main error log].The main error log is:{log}'''}, ]

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0.5)
    return response['choices'][0]['message']['content']


def askChatGPT_proposed(log, logs, solution):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    MODEL = 'gpt-3.5-turbo-16k'

    result = list()
    for i in range(len(logs)):
        dic = dict()
        dic['log'] = logs[i]
        dic['resolution'] = solution[i]
        result.append(dic)
    ''
    messages = [{"role": "system", "content": "You are an helpful assistant."}, {"role": "user",
                                                                                 "content": f'''I have encountered an error. 
                                                                                 I will provide you with [the main error log] and [several similar error logs and their solutions]. 
                                                                                 Some of these [several similar error logs and their solutions] may be helpful in resolving this issue or not. 
                                                                                 Several similar error logs and their solutions are listed below:{result}
                                                                              Notification: Combine your knowledge about the main error log with the key knowledge mentioned in the [several similar error logs and their solutions]. 
                                                                            You should only output the steps which are executable to fix [the main error log].The main error log is:{log}'''}, ]
    ''

    messages = [{"role": "system", "content": "You are an helpful assistant."}, {"role": "user",
                                                                                 "content": f'''My system output an error log. You should give the executable steps to fix the error log.
     I will provide you with the error log and some examples including [several similar error logs and their solutions]. 
     Some of these [several similar error logs and their solutions] may be helpful in resolving the error log or not. 
     Several similar error logs and their solutions:{result}
     The error log:{log}'''}, ]


    str =  f'''My system output an error log. You should give the executable steps to fix the error log.
     I will provide you with the error log and some examples including [several similar error logs and their solutions]. 
     Some of these [several similar error logs and their solutions] may be helpful in resolving the error log or not. 
     Several similar error logs and their solutions:{result}
     The error log:{log}'''



    # encoded_input=tokenizer(solution)
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0.5)
    return response['choices'][0]['message']['content']



def askChatGPT_proposed_gpt4(log, logs, solution):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    MODEL = 'gpt-4'

    result = list()
    for i in range(len(logs)):
        dic = dict()
        dic['log'] = logs[i]
        dic['resolution'] = solution[i]
        result.append(dic)
    ''
    messages = [{"role": "system", "content": "You are an helpful assistant."}, {"role": "user",
                                                                                 "content": f'''I have encountered an error. 
                                                                                 I will provide you with [the main error log] and [several similar error logs and their solutions]. 
                                                                                 Some of these [several similar error logs and their solutions] may be helpful in resolving this issue or not. 
                                                                                 Several similar error logs and their solutions are listed below:{result}
                                                                              Notification: Combine your knowledge about the main error log with the key knowledge mentioned in the [several similar error logs and their solutions]. 
                                                                            You should only output the steps which are executable to fix [the main error log].The main error log is:{log}'''}, ]
    ''

    messages = [{"role": "system", "content": "You are an helpful assistant."}, {"role": "user",
                                                                                 "content": f'''My system output an error log. You should give the executable steps to fix the error log.
     I will provide you with the error log and some examples including [several similar error logs and their solutions]. 
     Some of these [several similar error logs and their solutions] may be helpful in resolving the error log or not. 
     Several similar error logs and their solutions:{result}
     The error log:{log}'''}, ]


    str =  f'''My system output an error log. You should give the executable steps to fix the error log.
     I will provide you with the error log and some examples including [several similar error logs and their solutions]. 
     Some of these [several similar error logs and their solutions] may be helpful in resolving the error log or not. 
     Several similar error logs and their solutions:{result}
     The error log:{log}'''



    # encoded_input=tokenizer(solution)
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0.5)
    return response['choices'][0]['message']['content']



def get_response_proposed(log, title):
    logs, solutions = find_simlog.get_simlog_solution(log, title, 4)

    return askChatGPT_proposed(log,logs=logs, solution=solutions)

def get_response_proposed_gpt4(log, title):
    logs, solutions = find_simlog.get_simlog_solution(log, title, 4)

    return askChatGPT_proposed_gpt4(log,logs=logs, solution=solutions)


def get_response_log_description(log, description):
    return askChatGPT_log_description(log, description)


def get_response_lognoly(log):
    return askChatGPT_logonly(log=log)



def get_response_lognoly_gpt4(log):
    return askChatGPT_logonly_gpt4(log=log)