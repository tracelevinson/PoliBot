"""
clean_urls.py

Script to narrow large raw URL collection to only relevant URLs for final dataset. Program executes a set of regex rules specific to each news source and its respective article URL format. Rules are defined in the leaf_formats dictionary below.

Input: raw_urls.tsv (raw URLs accumulated in extract_urls.py)
Output: clean_urls.tsv (relevant URLs to be sent to scrape_url_text.py)
"""
###
from utils import *
###

leaf_formats = {
'cnn': 'cnn.com/[1-2][0-9][0-9][0-9]/[0-1]*[0-9]/[0-3]*[0-9]',
'fox_news': 'foxnews.com/politics/[1-2][0-9][0-9][0-9]/[0-1]*[0-9]/[0-3]*[0-9]',
'vox': 'vox.com/(?!.*videos).*[1-2][0-9][0-9][0-9]/[0-1]*[0-9]/[0-3]*[0-9]',
'ny_times': 'nytimes.com/[1-2][0-9][0-9][0-9]/[0-1]*[0-9]/[0-3]*[0-9]/[^(podcasts)]',
'wapo': 'washingtonpost.com/.*[1-2][0-9][0-9][0-9]/[0-1]*[0-9]/[0-3]*[0-9]',
'npr': 'www.npr.org/.*[1-2][0-9][0-9][0-9]/[0-1]*[0-9]/[0-3]*[0-9]',
'la_times': 'latimes.com/(?!sports)(?!espanol).*[1-2][0-9][0-9][6-8][0-1]*[0-9][0-3]*[0-9]',
'wsj':'wsj.com/articles',
'msnbc':'msnbc.com/msnbc/',
'bbc':'bbc.*[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]',
'politico':'politico.com/[^(video)].*[1-2][0-9][0-9][0-9][0-1]*[0-9][0-3]*[0-9]',
'reuters':'reuters.com/article',
'usa_today':'usatoday.com/story/[^(life)|(tech)|(sports)|(travel)|(weather)]',
'chicago':'chicagotribune.com/news.*[1-2][0-9][0-9][0-9][0-1]*[0-9][0-3]*[0-9]',
'newsweek':'newsweek.com/[^(search)|(privacy)|(terms)]'
}

# Import raw_urls file compiled in extract_urls.py
raw_urls = pd.read_csv(home + 'data/raw_urls.tsv', sep='\t', header=0)

# Restrict URLs to relevant articles determined by leaf_formats
combined_regex = re.compile('|'.join(x for x in list(leaf_formats.values())))
clean_urls = raw_urls[raw_urls['url'].str.contains(combined_regex)].drop_duplicates('url').sort_values(by=['source','url'])
clean_urls.reset_index(inplace=True, drop=True)

# Make miscellaneous fixes to URL list
clean_urls['url'] = clean_urls.url.str.replace('(\?mod=searchresults).*', '')
clean_urls['url'][34873] = 'https://www.politico.com/story/2018/01/30/trump-state-of-the-union-2018-speech-quotes-379314'
clean_urls = clean_urls[clean_urls.index!=43018] # drop disruptive link

# Output clean urls to TSV
clean_urls_path = home + 'data/clean_urls.tsv'
URLsToTSV(clean_urls, output=clean_urls_path, mode='w')
