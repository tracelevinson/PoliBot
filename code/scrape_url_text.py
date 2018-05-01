"""
scrape_url_text.py

Scrape body text from article URLs using xpaths specific to each source, specified in html_tags dictionary.

Input: clean_urls.tsv (URL strings of all articles to scrape)
Outputs: articles_raw.tsv (raw articles dataset), articles_full_detailed.tsv (clean articles dataset); both datasets include url, source, text
"""
###
from utils import *
import requests
###
raw_path = home + 'data/articles_raw.tsv'
final_path = home + 'data/articles_final.csv'
###

html_tags = {
'bbc': [{'title': 'h1', 'content': ['class', 'story-body__inner', 'p']}, {'title': 'h1', 'content': ['class', 'vxp-media__summary', 'p']}],
'chicago': [{'title': 'h1', 'content': ['class', 'trb_ar_page', 'p']}],
'cnn': [{'title': 'h1', 'content': ['class', 'zn-body__paragraph', None]}, {'title': 'h1', 'content': ['class', 'zn-body__paragraph', 'p']}, {'title': ['class', 'article-title speakable'], 'content': ['id', 'storytext', 'p']}],
'fox_news': [{'title': 'h1', 'content': ['class', 'article-body', 'p']}],
'la_times': [{'title': 'h1', 'content': ['class', 'collection collection-cards', 'p']}],
'msnbc': [{'title': 'h1', 'content': ['itemprop', 'articleBody', 'p']}],
'newsweek': [{'title': 'h1', 'content': ['class', 'article-body', 'p']}],
'npr': [{'title': 'h1', 'content': ['class', 'transcript storytext', 'p']}],
'ny_times': [{'title': 'h1', 'content': ['id', 'story', 'p']}, {'title': None, 'content': ['class', 'post-content', 'p']}],
'politico': [{'title': 'h1', 'content': ['class', 'story-text', 'p']}],
'reuters': [{'title': 'h1', 'content': ['class', 'body_1gnLA', 'p']}],
'usa_today': [{'title': 'h1', 'content': ['class', 'story primary-content', 'p']}, {'title': 'h1', 'content': ['class', 'content-well', 'p']}, {'title': 'h1', 'content': ['itemprop', 'mainEntity articleBody', 'p']}],
'vox': [{'title': 'h1', 'content': ['class', 'c-entry-content', 'p']}],
'wapo': [{'title': 'h1', 'content': ['class', 'article-body content-format-default', 'p']}, {'title': 'h1', 'content': ['id', 'article-body', 'p']}],
'wsj': [{'title': 'h1', 'content': ['class', 'article-wrap', 'p']}, {'title': 'h1', 'content': ['class', 'article-wrap', 'p']}, {'title': 'h1', 'content': ['class', 'djs-5t-item djs-5t-clear djs-5t-divider', 'p']}, {'title': 'h1', 'content': ['class', 'djs-5t-item djs-5t-clear djs-5t-divider', None]}]
}

# establish articles dataset
articles_df = pd.read_csv(home + 'data/clean_urls.tsv', sep='\t', header=0)
articles_df['text'] = ''

# Establish transition from main URL extraction script to WSJ script
wsj_start = min(articles_df[articles_df.source=='wsj'].index)

# Extract text from all sources except WSJ (no login required)
for i in range(wsj_start):

    # Define variables
    url = articles_df.url[i]
    source = articles_df.source[i]

    # Access article webpage
    page = requests.get(articles_df.url[i], headers={'User-Agent':'Mozilla/5.0'})

    # Verify web connection
    if page.status_code!=200:
        print('Page not downloaded properly: ' + str(i) +', '+url)
        error_urls.append(url)
        with open(home+'data/error_urls.txt','a') as f:
            f.write(str(i)+': '+url+'\n')
        continue

    # Navigate through HTML tree using html_tags specs
    soup = BeautifulSoup(page.content, 'html.parser')

    # Extract article text using BeautifulSoup
    # Note: CNN articles are handled separately, as their content is gathered from multiple elements within its html_tags dictionary.
    if source=='cnn':
        articles_df.text[i] = ExtractCNNText(soup, i, url, html_tags[source])
    else:
        articles_df.text[i] = ExtractURLText(soup, i, url, source, html_tags[source])

    # Print warning for blank articles
    if articles_df.text[i]=='':
        print('No text collected for ' + str(i) + ', ' + url)

    # Monitor progress and update TSV file every 100 articles
    if i==0:
        with open(raw_path, 'w') as f:
            f.write('index\turl\tsource\ttext\n')
    elif i%100==0 or i==wsj_start-1:
        print(i, source)
        articles_df[articles_df.text!=''].tail(100).to_csv(raw_path, sep='\t', header=False, mode='a')

# Extract text from WSJ articles (login required)
articles_df = ExtractWSJText(df=articles_df, start_index=wsj_start, html_tags=html_tags, path=raw_path)

# Remove extraneous obs and prepare text for embeddings model
articles_df = articles_df.drop_duplicates(['url']).dropna(axis=0, how='any')
articles_df['text'] = articles_df.text.apply(process_text)

# Print article count and median word count for each source in final dataset
df_copy = articles_df
df_copy.text = df_copy.text.str.replace('[^A-Za-z0-9 ]', '').str.split().apply(len)
print(pd.concat([df_copy.source.value_counts(),
                 df_copy.groupby(['source'])['text'].median().astype(int)],axis=1)
        .rename(index=str, columns={'source': 'num_articles', 'text': 'median_word_count'}))
del df_copy

# Save finalized articles dataset ready for modeling
articles_df.to_csv(final_path, index=False)
