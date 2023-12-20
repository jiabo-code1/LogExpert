import os
from bs4 import BeautifulSoup
import re
import nltk
import csv
import json
from string import punctuation



def get_specific_text(soup):
    text = ''
    for element in soup.contents:
        if element.name:
            #if element.name != 'pre' and element.name != 'code' and element.name != 'blockquote':
            if element.name != 'code' and element.name != 'blockquote':
                text += element.text
        else:
            text += element.text
    return text.strip()


def extract(input_file_path, output_file_path):
    file_list = os.listdir(input_file_path)
    csv_file = open(output_file, 'w', encoding='utf-8')
    writer = csv.writer(csv_file)
    writer.writerow(['description','tags','resolution', 'log'])
    for page_index in range(1, len(file_list)):


        page_file = open(input_file_path + str(page_index) + '.html', encoding='utf-8')
        page_soup = BeautifulSoup((page_file.read().strip()), 'html.parser')
        page_file.close()

        try:
            title = page_soup.find('div', {'id': 'question-header'}).find('h1').text.strip()
        except AttributeError:
            title = page_soup.find('div', {'id': 'question'}).find('h2').text.strip()

        question = page_soup.find('div', {'id': 'question'}).find('div', {'class': 's-prose js-post-body'})
        nlp_description = get_specific_text(question)
        #nlp_description=question.text.strip()
        row_data = []
        row_data.append('title:' + title)
        row_data.append('description:' + nlp_description)



        tag_links = page_soup.find_all('a', {'class': 'post-tag'})

        tags = ''
        for tag_link in tag_links:
            tag = tag_link.text.strip()
            tags = tags + tag + "|"
        row_data.append('tags:'+tags)


        try:
            answers = page_soup.find('div', {'id': 'answers'}).find_all('div', {'class': 's-prose js-post-body'})
        except AttributeError:
            answers = []
            answer_soups = page_soup.find_all('div', {'class': '-summary answer'})
            for answer_soup in answer_soups:
                answers.append(answer_soup.find('div', {'class': 's-prose js-post-body'}))
        for answer_index in range(0, len(answers)):
            answer = answers[answer_index].get_text()
            row_data.append('resolution:' + answer)
            break


        alternative_logs1 = question.find_all('code')
        alternative_logs2 = question.find_all('blockquote')
        for log in alternative_logs1:
            log_text = log.text.strip()
            # if is_log(log_text, get_log_above_text(log), log_extract_thesaurus_path):
            if True:
                row_data.append('log:' + log_text)

        for log in alternative_logs2:
            log_text = log.text.strip()
            if True:
                row_data.append('log:' + log_text)

        writer.writerow(row_data)
    csv_file.close()


extract(input_file_path, output_file)
