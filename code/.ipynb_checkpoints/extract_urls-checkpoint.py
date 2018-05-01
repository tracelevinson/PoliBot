"""
extract_urls.py

Script to extract text from article URLs using lynx. Navigates to news source home page, or a higher-level page containing many embedded articles, and spits out all embedded article URLs to a text file for easy importation to Python.

Input: URL string (string)
Output: Article's text body (string)
"""

import os, csv
import pandas as pd

home_dir = '/Users/tracelevinson/Documents/DataScience/PoliBot/'

source_url_df = pd.read_csv(home_dir + 'data/source_urls.csv')

for url in source_url_df['url']:
    print(os.system('lynx -dump ' + url))