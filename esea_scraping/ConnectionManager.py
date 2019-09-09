# -*- coding: utf-8 -*-

# Adapted to Python3 from
# https://jarroba.com/anonymous-scraping-by-tor-network/

__author__ = 'RicardoMoya'


import time
from urllib.error import HTTPError
from urllib.request import build_opener, install_opener, Request, urlopen, ProxyHandler
from stem import Signal
from stem.control import Controller
from procUserAgentList import getUAlist, DATA_PATH, FILENAME, SKIP_UA_CAT, SKIP_UA
import numpy as np

PORT=9051
TOR_HOST= "192.168.2.240"
PROXY_ADDR = "192.168.2.240:8118"

IP_SHOWING_HOST = "http://icanhazip.com"
CHARSET= 'UTF-8'
ACCEPT = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'
ACCEPT_ENCODING = 'gzip, deflate, br'
ACCEPT_LANG = 'en-US,en;q=0.9'
CACHE_CONTROL = 'max-age=0'

UA_LIST = getUAlist(DATA_PATH, FILENAME, SKIP_UA_CAT, SKIP_UA)

#from https://linkminer.com/
REF_LIST = [
  'http://www.dust2.dk/Profil/KMcs',
  'https://www.reddit.com/r/GlobalOffensive/comments/5jzw9z/bogdan/',
  'http://google.com/',
  'https://www.twitch.tv/esl_csgo',
  'https://en.wikipedia.org/wiki/Team_Fortress_2',
  'https://pro.eslgaming.com/csgo/proleague/',
  'https://www.intel.com/content/www/us/en/gaming/esports.html',
  'https://www.8bitgaming.de/',
  'https://www.needforseatusa.com/en_us/partners',
  'https://www.8bitgaming.de/tmp/rss.xml',
  'https://www.8bitgaming.de/index.php',
  'https://www.8bitgaming.de/index.php?site=news',
  'http://pro.eslgaming.com/csgo/proleague/finals-7/',
  'http://www.teamplayergaming.com/content/',
  'https://www.needforseatusa.com/partners',
  'http://www.bailopan.net/csdm/index.php?page=credits',
  'https://pro.eslgaming.com/odense/',
  'http://www.teamplayergaming.com/forum.php',
  'http://www.teamplayergaming.com/content/626-approaching-infinity-review.html',
  'http://www.esportsea.com/',
  'https://liquipedia.net/counterstrike/ESL/Pro_League/Season_8/Europe',
  'https://liquipedia.net/counterstrike/ESL/Pro_League/Season_8/Finals',
  'https://liquipedia.net/counterstrike/ESL/Pro_League/Season_8/North_America',
  'http://www.teamplayergaming.com/db4d-public-discussion/',
  'http://www.teamplayergaming.com/content/209-game-servers.html',
  'http://www.teamplayergaming.com/tcz-public-discussion/',
  'http://www.teamplayergaming.com/content/229-tpg-affiliate-links.html',
  'http://www.teamplayergaming.com/competitions.php',
  'http://www.teamplayergaming.com/alpha-wolves-public-discussion/',
  'http://www.teamplayergaming.com/faq.php',
  'http://www.teamplayergaming.com/content/366-tpg-privacy-policy.html',
  'http://forum.dawahfrontnigeria.com/syndication.php?type=atom1.0&fid=4',
  'https://www.8bitgaming.de/index.php?site=news&show=8 Bit Gaming',
  'http://www.teamplayergaming.com/pc-hardware-and-technology/84214-anandtech-news.html',
  'https://lifehacker.ru/kibersportsmen/',
  'http://www.gameboysbook.com/',
  'https://kotaku.com/counter-strike-player-banned-for-1000-years-for-alleged-1792108179',
  'https://www.gpforums.co.nz/threads/502805-New-to-CSGO-amp-CSNZ?p=11024483',
  'https://www.8bitgaming.de/index.php?site=news.php',
  'https://www.topcsgobettingsites.com/william-hill/',
  'https://ewave-esports.de/',
  'https://www.golem.de/news/esea-1-5-millionen-datensaetze-von-e-sportlern-im-netz-1701-125467.html',
  'https://www.gpforums.co.nz/threads/502805-New-to-CSGO-amp-CSNZ',
  'https://www.sweclockers.com/forum/trad/1547758-jag-moter-fuskare-varenda-game-mg1-mm-hur-loser-man-detta-elohell',
  'http://www.teamplayergaming.com/content/section/511-games/',
  'http://www.teamplayergaming.com/d-i-c-k-s-public-discussion/',
  'http://www.teamplayergaming.com/content/section/455-chivalry/',
  'http://www.teamplayergaming.com/content/section/456-counter-strike/',
  'http://www.teamplayergaming.com/content/section/457-dayz/',
  'http://www.teamplayergaming.com/content/section/458-left-4-dead/',
  'http://www.teamplayergaming.com/content/section/459-team-fortress/',
  'http://www.teamplayergaming.com/content/section/461-day-defeat/',
  'http://www.teamplayergaming.com/content/section/462-battlefield-3-a/',
  'http://www.teamplayergaming.com/content/section/463-natural-selection-2-a/',
  'http://www.teamplayergaming.com/content/section/493-one-on-one/',
  'http://www.teamplayergaming.com/content/section/494-podcast/'
]

class ConnectionManager:
  def __init__(self):
    self.new_ip = "0.0.0.0"
    self.old_ip = "0.0.0.0"
    self.new_identity()

  @classmethod
  def _get_connection(self):
    """
    TOR new connection
    """
    with Controller.from_port(address= TOR_HOST, port=PORT) as controller:
      controller.authenticate(password="csgoscraping")
      #print(controller.get_version())
      #print(controller.is_newnym_available())
      #print(controller.get_newnym_wait())
      controller.signal(Signal.NEWNYM)
      controller.close()

  @classmethod
  def _set_url_proxy(self):
    """
    Request to URL through local proxy
    """
    proxy_support = ProxyHandler({"http": PROXY_ADDR})

    opener = build_opener(proxy_support)
    install_opener(opener)


  @classmethod
  def request(self, url, referer=None, cookie=None):
    """
    TOR communication through local proxy
    :param url: web page to parser
    :return: request
      """
    try:
      print('Connection URL:', url)
      self._set_url_proxy()
      ua = np.random.choice(UA_LIST, size=1)[0]
      if referer is None:
        referer = np.random.choice(REF_LIST, size=1)[0]
      print('@@@@@@@', ua)
      param_dict = {
        'User-Agent': ua,
        'Accept-Charset' : CHARSET,
        'referer' : referer,
        'accept' : ACCEPT,
        # Don't use ACCEPT_ENCODING unless your are prepared to
        # decode a compressed content
        #'accept-encoding' : ACCEPT_ENCODING,
        'accept-language' : ACCEPT_LANG,
        'cache-control' : CACHE_CONTROL,
        'upgrade-insecure-request' : 1
      }
      if cookie is not None:
        param_dict['cookie'] = cookie

      print(param_dict)

      request = Request(url, None, param_dict)
      request = urlopen(request)
      return request

    except HTTPError as e:
      return e.message

  def new_identity(self):
    """
    new connection with new IP
    """
    # First Connection
    if self.new_ip == "0.0.0.0":
      self._get_connection()
      self.new_ip = self.request(IP_SHOWING_HOST).read().strip()
    else:
      self.old_ip = self.new_ip
      self._get_connection()
      self.new_ip = self.request(IP_SHOWING_HOST).read().strip()

    seg = 0

    print('new IP', self.new_ip, 'old IP', self.old_ip)

    # If we get the same ip, we'll wait 5 seconds to request a new IP
    while self.old_ip == self.new_ip:
      time.sleep(5)
      seg += 5
      print ("Waiting to obtain new IP: %s Seconds" % seg)
      self.new_ip = self.request(IP_SHOWING_HOST).read()

    print ("New connection with IP: %s" % self.new_ip)

