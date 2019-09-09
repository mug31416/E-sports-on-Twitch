# -*- coding: utf-8 -*-

# Adapted to Python3 from
# https://jarroba.com/anonymous-scraping-by-tor-network/

__author__ = 'RicardoMoya'


import time
import re, sys
from stem import Signal
from stem.control import Controller

from selenium import webdriver
from selenium.webdriver.common.proxy import ProxyType

PORT=9051
TOR_HOST= "192.168.2.240"
PROXY_HOST = "192.168.2.240"
PROXY_PORT = 8118
PROXY_ADDR = f"{PROXY_HOST}:{PROXY_PORT}"

IP_SHOWING_HOST = "https://icanhazip.com"


class ConnectionManagerJS:

  def __init__(self):
    self.driver = None
    self.new_session()

    self.new_ip = "0.0.0.0"
    self.old_ip = "0.0.0.0"

  @classmethod
  def _get_connection(self):
    """
    TOR new connection
    """
    with Controller.from_port(address= TOR_HOST, port=PORT) as controller:
      controller.authenticate(password="csgoscraping")
      controller.signal(Signal.NEWNYM)
      controller.close()

  def send_request(self, url):
    """
    TOR communication through local proxy
    :param url: web page to parser
    :return: request
      """

    self.driver.refresh()
    print('Connection URL:', url)
    self.driver.get(url)


  def new_session(self):

    self.shutdown()
    self.driver = webdriver.Firefox()

    #self.driver.delete_all_cookies()

  def extract_ip(self):

    html = self.extract_html()
    return re.sub(r'</?[^>]*>', '', html).strip()


  def extract_html(self):
    #self.driver.add_cookie({'name': 'foo', 'value': 'bar'})
    return self.driver.page_source


  def new_identity(self):
    """
    new connection with new IP
    """

    # First Connection
    if self.new_ip == "0.0.0.0":
      self._get_connection()
      self.send_request(IP_SHOWING_HOST)
      print(self.extract_html())
      self.new_ip = self.extract_ip()

    else:
      self.old_ip = self.new_ip
      self._get_connection()
      self.send_request(IP_SHOWING_HOST)
      self.new_ip = self.extract_ip()

    seg = 0

    print('new IP', self.new_ip, 'old IP', self.old_ip)

    # If we get the same ip, we'll wait 5 seconds to request a new IP
    while self.old_ip == self.new_ip:

      time.sleep(5)
      seg += 5
      print ("Waiting to obtain new IP: %s Seconds" % seg)

      self.new_session()
      self.send_request(IP_SHOWING_HOST)
      self.new_ip = self.extract_ip()


    print ("New connection with IP: %s" % self.new_ip)

  def shutdown(self):
    if self.driver is not None:
      self.driver.close()

