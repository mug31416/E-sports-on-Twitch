import xml.etree.ElementTree as ET
import os
import numpy as np

DATA_PATH = '/Users/anna/Courses/LargeMultiMedia/project/data/'
FILENAME = 'user_agent.xml'
SKIP_UA_CAT = ['Spiders - Search', 'Miscellaneous', 'UA List :: About', 'Mobile Devices']
SKIP_UA = ['Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)']

def getUAlist(DATA_PATH,FILENAME,SKIP_UA_CAT,SKIP_UA):

  tree = ET.parse(os.path.join(DATA_PATH, FILENAME))
  root = tree.getroot()

  list = []

  for child in root:
    print(child.attrib)

    desc = child.attrib.get('description')

    if desc in SKIP_UA_CAT:
      continue

    for sub in child:
      ua = sub.attrib.get('useragent')

      if ua is None:
        continue

      if len(ua) == 0:
        continue

      if ua in SKIP_UA:
        continue

      list.append(ua)

  return np.array(list)


if __name__=="__main__":

  list = getUAlist(DATA_PATH, FILENAME, SKIP_UA_CAT, SKIP_UA)
  print(list)

