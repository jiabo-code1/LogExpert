#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import numpy
import scrapy
from stackoverflow.spiders.items import StackoverflowItem
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('monitor')
logger.setLevel(logging.INFO)

fh = logging.FileHandler('monitor.log')
fh.setLevel(logging.INFO)

fh.setFormatter(formatter)
logger.addHandler(fh)


class StackoverflowSpider(scrapy.Spider):

    name = "stackoverflow"

    l1 = numpy.arange(1000)
    l2 = numpy.zeros(1000)
    li = numpy.array([l1, l2])

    def start_requests(self):
        link_file = open("../link.txt", 'r')
        urls = link_file.readlines()

        for url in urls:
            print(url)
            n = urls.index(url)
            yield scrapy.Request(url=url, callback=self.parse,dont_filter=True)

    def parse(self, response, li=li):
        print("--------------------")
        for i in range(0, 10000):
            if li[1][i] == 0:
                num = li[0][i]
                li[1][i] = 1
                break
        num=num+1
        f = open(
            "D://stackoverflow_apache_real_pages//" + str(int(
                num)) + '.html',
            'w',encoding='UTF-8')
        f.write(response.text)

