from bs4 import BeautifulSoup
import requests

content = requests.get('https://leetcode.com/problemset/all/', timeout=60).content

# lxml is apparently faster than other settings.
soup = BeautifulSoup(content, "lxml")