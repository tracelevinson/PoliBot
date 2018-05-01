"""
extract_urls.py

Script to extract all relevant article URLs from news source websites. Navigates to news source home page, or a higher-level search page containing many resulting articles, and spits out all embedded URLs to a text file for easy importation in Python.

Input: source_urls.csv (source pages to scrape URLS from)
Output: raw_urls.tsv (set of all raw URLs collected from each news source)
"""
###
from utils import *
###
raw_urls_path = home + 'data/raw_urls.tsv'
###

# Create dictionary to reference URL stems from each news source
sources = pd.read_csv(home + 'data/source_urls.csv', header=0)
sources = {sources['source'][i]: sources['url'][i] for i in sources.index}

print('Extracting URLs:')

# CNN
print(' (1/15) CNN...')
cnn_urls = []
for i in range(1,10001,100):
    try:
        cnn_json = urllib.request.urlopen(sources['cnn'].format(i)).read()
        cnn_data = json.loads(cnn_json)
        for article in cnn_data['result']:
            cnn_urls.append(article['url'])
    except:
        pass
URLsToTSV(cnn_urls, source_name='cnn', output=raw_urls_path, mode='w')

# Fox News
print(' (2/15) Fox News...')
fox_urls = URLsFromLynx(sources['fox_news'], min_page=1296, max_page=2296)
URLsToTSV(fox_urls, source_name='fox_news', output=raw_urls_path)

# Vox
print(' (3/15) Vox...')
vox_urls = []
for i in range(1,4):
    vox_urls.extend(URLsFromXPath(sources['vox'].format(i), max_page=100, class_name='c-archives-load-more__button'))
URLsToTSV(vox_urls, source_name='vox', output=raw_urls_path)

# New York Times
print(' (4/15) New York Times...')
# Load archived articles by hitting "Load More" button max_page times
# Note: this will take ~10 minutes to load.
nyt_urls = URLsFromXPath(sources['ny_times'], max_page=500, class_name='Search-showMoreWrapper--1Z88y', sleep=2)
URLsToTSV(nyt_urls, source_name='ny_times', output=raw_urls_path)

# Washington Post
print(' (5/15) Washington Post...')
# Load archived articles by hitting "Load More" button max_page times
wapo_politics_urls = URLsFromXPath(sources['wapo_politics'], max_page=100, class_name='pb-loadmore')
wapo_security_urls = URLsFromXPath(sources['wapo_security'], max_page=100, class_name='pb-loadmore')
URLsToTSV(wapo_politics_urls + wapo_security_urls, source_name='wapo', output=raw_urls_path)

# NPR
print(' (6/16) NPR...')
npr_urls = URLsFromLynx(sources['npr'], max_page=7516, step=15)
URLsToTSV(npr_urls)

# Los Angeles Times
print(' (7/15) Los Angeles Times...')
la_trump_urls = URLsFromLynx(sources['la_times_trump'], max_page=100)
la_natpol_urls = URLsFromLynx(sources['la_times_natpol'], max_page=20)
URLsToTSV(la_trump_urls + la_natpol_urls, source_name='la_times', output=raw_urls_path)

# Wall Street Journal
print(' (8/15) Wall Street Journal...')
wsj_urls = URLsFromLynx(sources['wsj'], max_page=500)
URLsToTSV(wsj_urls, source_name='wsj', output=raw_urls_path)

# MSNBC
print(' (9/15) MSNBC...')
msnbc_urls = URLsFromXPath(sources['msnbc'], max_page=500)
URLsToTSV(msnbc_urls, source_name='msnbc', output=raw_urls_path)

# BBC
print(' (10/15) BBC...')
bbc_urls = URLsFromXPath(sources['bbc'], max_page=50)
URLsToTSV(bbc_urls, source_name='bbc')

# Politico
print(' (11/15) Politico...')
politico_urls = URLsFromLynx(sources['politico'], max_page=500)
URLsToTSV(politico_urls, source_name='politico', output=raw_urls_path)

# Reuters
print(' (12/15) Reuters...')
reuters_urls = URLsFromLynx(sources['reuters'], max_page=500)
URLsToTSV(reuters_urls, source_name='reuters', output=raw_urls_path)

# USA Today
print(' (13/15) USA Today...')
usa_urls = URLsFromLynx(sources['usa_today'], max_page=500)
URLsToTSV(usa_urls, source_name='usa_today', output=raw_urls_path)

# Chicago Tribune
print(' (14/15) Chicago Tribune...')
chicago_urls = URLsFromXPath(sources['chicago'], max_page=500)
URLsToTSV(chicago_urls, source_name='chicago', output=raw_urls_path)

# NewsWeek
print(' (15/15) NewsWeek...')
nw_urls = URLsFromLynx(sources['newsweek'], max_page=500)
URLsToTSV(nw_urls, source_name='newsweek', output=raw_urls_path)
