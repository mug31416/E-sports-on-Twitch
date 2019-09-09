# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

#TARGET_PAGE = "http://searchivarius.org/about"
#TARGET_PAGE = "http://rbc.ru"
TARGET_PAGE = "http://icanhazip.com"

from ConnectionManager import ConnectionManager, CHARSET

cm = ConnectionManager()

for j in range(5):
  for i in range(3):
    print ("\t\t" + cm.request(TARGET_PAGE).read().decode(CHARSET))
    cm.new_identity()