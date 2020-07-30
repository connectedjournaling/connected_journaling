import requests 
import urllib.request
import time, types
from bs4 import BeautifulSoup
import csv, sys

""" Gaining access to the website to scrape from """
website = "https://www.my-diary.org"
url = sys.argv[1]
response = requests.get(url)
#time.sleep(1) #pause the code for a sec
print (response) #Respnse 200- Access granted

""" Initial data storages """
soup = BeautifulSoup(response.text, 'html.parser')
all_links = soup.findAll('a')
CSV_name = "./diary#1.csv"
all_sources = [] 
paragraph_list = [['title', 'text']]

""" Getting all the links of diary entires on the website """
for link in all_links:
	source = link['href']
	if source[:5] == '/read': #Journal entries start with read
		all_sources.append(website + source)



""" From all sources, extract journal entries """
for source in all_sources: 
	response = requests.get(source)
	soup = BeautifulSoup(response.text, 'html.parser')
	container = soup.find('main', attrs={'class':'container'})


	count = 0
	print (source)
	journal_title = source[source.rindex('/')+1:]
	paragraph = [journal_title]

	have = []

	for x in container.find_all('p'):
		if x.text != "": 
			count += 1
			if count > 2: # This is to skip standard garbage on site
				text = x.text
				text = text.replace(',', "")
				# 'Login' signifies end of journal entry
				#  Whenever we find the end of a journal entry, we take all the text. 
				#  If we don't find the end of the journal entry, we move to the next paragraph. 

				try: 
					if text.index('Login') > -1: # Found Login
						break_index = text.index('Login') #Take text up until Login
						if len(paragraph) == 2:
							paragraph[1] = paragraph[1] + ' ' + text[:break_index]
						else: 
							paragraph.append(text[:break_index])
						break

				except: 
					if len(paragraph) == 2: 
						paragraph[1] = paragraph[1] + ' ' + text
					else: 
						paragraph.append(text)


	paragraph_list.append(paragraph)


""" Writing list of lists to CSV file """
with open(CSV_name, "w", newline="") as f:
	writer = csv.writer(f)
	writer.writerows(paragraph_list)