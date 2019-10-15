import urllib.request
from bs4 import BeautifulSoup

wikiURL = "https://scikit-learn.org/stable/modules/clustering.html#clustering"
fpURL = urllib.request.urlopen(wikiURL)

# Assigning Parsed Web Page into a Variable
soup = BeautifulSoup(fpURL, "html.parser")

table = soup.find('table')
table_rows = table.find_all('tr')

for tr in table_rows:
    td = tr.find_all('td')
    row = [i.text for i in td]
    print(row)