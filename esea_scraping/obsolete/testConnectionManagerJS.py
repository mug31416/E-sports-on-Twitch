# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

cm = None

def shutdown():
  if cm is not None: cm.shutdown()

import atexit
atexit.register(shutdown)

#TARGET_PAGE = "http://searchivarius.org/about"
#TARGET_PAGE = "http://rbc.ru"
#TARGET_PAGE = "https://icanhazip.com"
#TARGET_PAGE = "https://play.esea.net"

from ConnectionManagerJS import ConnectionManagerJS

cm = ConnectionManagerJS()
cm.new_identity()

# for j in range(2):
#   for i in range(2):
#     cm.send_request(TARGET_PAGE)
#     print ("\t\t" + cm.extract_html())
#     cm.new_identity()


#cm.send_request("https://www.whatismybrowser.com/")

cm.send_request("https://icanhazip.com")

#ext = input("input something to continue")

#cm.send_request("https://play.esea.net/users/2197239")




