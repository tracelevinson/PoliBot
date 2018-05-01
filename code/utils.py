"""
utils.py

Auxiliary file containing utility functions for extracting, manipulating, and reformatting data.

Input: None
Output: Importable utility functions
"""
###
import os, time, subprocess
from selenium import webdriver
import chromedriver_binary
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
###
import re
import nltk
nltk.download('punkt') # if not already downloaded
nltk.download('stopwords') # if not already downloaded
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
###
import pandas as pd
import numpy as np
import pickle
###
home = '' # set user home directory
source_display_map = {'bbc': 'BBC', 'chicago': 'Chicago Tribune', 'cnn': 'CNN', 'fox_news': 'Fox News', 'la_times': 'Los Angeles Times', 'msnbc': 'MSNBC', 'newsweek': 'NewsWeek', 'npr': 'NPR', 'ny_times': 'New York Times', 'politico': 'Politico', 'reuters': 'Reuters', 'usa_today': 'USA Today', 'vox': 'Vox', 'wapo': 'Washington Post', 'wsj': 'Wall Street Journal'}
###


def URLsFromXPath(source_url, max_page, class_name=None, sleep=1):
    """ Pull URLs from source code via HREF xpaths. If class_name is specified, the script hits the "Load More" button on a site max_page times. Otherwise, max_page is used to iterate through URL suffixes. """

    # Initiate Selenium webdriver instance
    driver = webdriver.Chrome()
    wait = WebDriverWait(driver, 10)

    urls_list = []

    # Vox, NY Times, Washington Post
    if class_name:

        driver.get(source_url)
        time.sleep(10)

        for i in range(max_page):
            try:
                more_button = wait.until(EC.presence_of_element_located((By.CLASS_NAME, class_name))).click()
                if i%10==0: print(i) # monitor progress every 10 loads
                if i%50==0:
                    time.sleep(5) # wait longer to recoup web stability every 50 loads
                else:
                    time.sleep(sleep) # wait 'sleep' secs for "Load More" button to repopulate
            except:
                break

        time.sleep(5) # wait for final page to load all HREFs

        urls_list.extend(ExtractHREFs(driver))

    # MSNBC, BBC, Chicago Tribune
    else:
        for i in range(max_page):
            try:
                driver.get(source_url.format(i))
                urls_list.extend(ExtractHREFs(driver))
            except:
                break

    driver.quit()

    return urls_list


def URLsFromLynx(source_url, max_page, min_page=1, step=1):
    """ Pull article URL list using Lynx to dump all embedded links in source_url. """

    result = []
    for i in range(min_page, max_page+1, step):
        try:
            lynx = subprocess.Popen('lynx -dump -listonly -unique_urls \"' + source_url.format(i) + '\"', shell=True, stdout=subprocess.PIPE).stdout.read()
            urls_str = re.sub('\\\\n','',str(lynx)[1:-1]) # [1:-1] to remove outside apostrophes produced by Lynx
            urls_list = re.compile('[ ]+[0-9]+.[ ]').split(urls_str)[1:] # create url list
            result.extend(urls_list)
        except:
            break

    return result


def ExtractHREFs(driver):
    """ Extract all HREF tags (HTML links) from website source code. """

    urls_list = []

    for result in driver.find_elements_by_xpath('//a[@href]'):
        article_url = result.get_attribute('href')
        urls_list.append(article_url)

    return urls_list


def URLsToTSV(source_urls, output, source_name=None, mode='a'):
    """ Add source URLs to full dataset compilation. """

    with open(output, mode) as url_file:
        if mode=='w':
            url_file.write('url\tsource\n')
        # raw_urls
        if source_name:
            for url in source_urls:
                url_file.write(str(url)+'\t'+str(source_name)+'\n')
        # clean_urls
        else:
            for i in source_urls.index:
                url_file.write(source_urls['url'][i]+'\t'+source_urls['source'][i]+'\n')


def ExtractURLText(soup, i, url, source, html_tag_list):
    """ Extract body text from URL. """

    for tags in html_tag_list:
        if not soup.find('div', {tags['content'][0]: tags['content'][1]}):
            if tags==html_tag_list[-1] and source=='ny_times':
                try:
                    region = soup.find_all('p', {'class':'story-body-text story-content'})
                    content = ' '.join([x.get_text() for x in region])
                    if tags['title']:
                        title = soup.find(tags['title']).get_text()
                        result_text = title + '. ' + content
                    else:
                        result_text = content
                except:
                    continue
            elif tags==html_tag_list[-1] and source=='wsj':
                try:
                    title = soup.find(tags['title']).get_text()
                    region = soup.find('ul', {'class':'djs-5t-list'}).find_all('p')
                    content = ' '.join([x.get_text() for x in region])
                    result_text = title + '. ' + content
                except:
                    continue
            else:
                continue
        else:
            # title
            if isinstance(tags['title'], str):
                try:
                    title = soup.find(tags['title']).get_text()
                except AttributeError:
                    if tags==html_tag_list[-1]:
                        write_to_errors(i, url)
                    continue
            elif tags['title']:
                try:
                    title = soup.find('div', {tags['title'][0]:tags['title'][1]}).get_text()
                except AttributeError:
                    if tags==html_tag_list[-1]:
                        write_to_errors(i, url)
                    continue
            # content
            if tags['content'][2]:
                try:
                    region = soup.find('div', {tags['content'][0]: tags['content'][1]}).find_all(tags['content'][2])
                except AttributeError:
                    if tags==html_tag_list[-1]:
                        write_to_errors(i, url)
                    continue
            else:
                try:
                    region = soup.find_all('div', {tags['content'][0]: tags['content'][1]})
                except AttributeError:
                    if tags==html_tag_list[-1]:
                        write_to_errors(i, url)
                    continue

            content = ' '.join([x.get_text() for x in region])
            if tags['title']:
                result_text = title + '. ' + content
            else:
                result_text = content
            return result_text

    return ''


def ExtractCNNText(soup, i, url, html_tag_list):
    """ Extension of ExtractURLText specialized for CNN articles (needed for spanning multiple xpaths to accrue body text). """

    title   = ''
    content = ''

    for tags in html_tag_list:
        if not soup.find('div', {tags['content'][0]: tags['content'][1]}):
            if tags==html_tag_list[-1] and (title + content != ''):
                return title + '. ' + content
            else:
                continue
        else:
            # title
            if isinstance(tags['title'], str):
                try:
                    if title == '':
                        title = soup.find(tags['title']).get_text()
                except AttributeError:
                    if tags==html_tag_list[-1]:
                        write_to_errors(i, url)
                    continue
            else:
                try:
                    if title == '':
                        title = soup.find('div', {tags['title'][0]:tags['title'][1]}).get_text()
                except AttributeError:
                    if tags==html_tag_list[-1]:
                        write_to_errors(i, url)
                    continue
            # content
            if tags['content'][2]:
                try:
                    region = soup.find('div', {tags['content'][0]: tags['content'][1]}).find_all(tags['content'][2])
                except AttributeError:
                    if tags==html_tag_list[-1]:
                        write_to_errors(i, url)
                    continue
            else:
                try:
                    region = soup.find_all('div', {tags['content'][0]: tags['content'][1]})
                except AttributeError:
                    if tags==html_tag_list[-1]:
                        write_to_errors(i, url)
                    continue

            content += ' '.join([x.get_text() for x in region])

    if title + content != '':
        return title + '. ' + content
    return ''


def ExtractWSJText(df, start_index, html_tags, path):
    """ Extension of ExtractURLText specialized for WSJ articles (needed for login requirement). """

    articles_df = df

    # Enter WSJ login information
    WSJ_EMAIL = ''
    WSJ_PASSWORD = ''

    # Initiate Selenium webdriver instance
    driver = webdriver.Chrome()
    wait = WebDriverWait(driver, 10)

    # Access URL and input login info
    driver.get(articles_df.url[start_index])
    login = driver.find_element_by_link_text("Sign In").click()
    user = driver.find_element_by_id("username").send_keys(WSJ_EMAIL)
    password = driver.find_element_by_id("password").send_keys(WSJ_PASSWORD)
    sign_in_button = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'basic-login-submit'))).click()
    time.sleep(5)

    # Iterate through WSJ articles and store text to articles_df.
    for i in range(start_index, len(df)):

        url = articles_df.url[i]
        source = 'wsj'
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'lxml')
        articles_df.text[i] = ExtractURLText(soup, source, html_tags[source])

        if articles_df.text[i]=='':
            print('No text collected for ' + str(i) + ', ' + url)

        # Update TSV file every 100 articles
        if i%100==0 or i==len(df):
            print(i, source)
            articles_df[articles_df.text!=''].tail(100).to_csv(path, sep='\t', header=False, mode='a')

    driver.quit()

    return articles_df


def write_to_errors(obs, url):
    """ Record obs in error_urls text file. """

    with open(home+'data/error_urls.txt','a') as f:
        f.write(str(obs)+': '+url+'\n')


def process_text(text):
    """ Process text to be analyzed by embeddings. """

    re_drop_symbols = '[^A-Za-z0-9 \t+_#$%,;:.\?\!@\-\'\"{}\[\]()\|/]'
    text = re.sub(re_drop_symbols, '', text)
    tokenized = ' '.join([w for w in word_tokenize(text) if str.lower(w) not in stopwords.words('english')])
    output = re.sub('[^A-Za-z0-9 -]','',str.lower(tokenized))

    return output


def vectorize_text(text, embed_dict, dim):
    """ Convert user query to embeddings vector using gensim embeddings model. """

    query_as_list = [x for x in process_text(text).split() if x in embed_dict]

    if len(query_as_list) == 0:
        return np.zeros(dim)

    return np.mean(embed_dict[query_as_list], axis=0)


def unpickle(filepath):
    """ Load pickle file into Python. """

    with open(filepath, 'rb') as f:
        return pickle.load(f)


def prune_sources(query, article_ranker):
    """ Remove user-specified sources from recommendation pool. """

    source_map = article_ranker.source_map

    if sum([int(src) not in source_map for src in query.split(',')])>0:
        return 'VALUE_ERROR'

    source_list = [source_map[int(s)] for s in query.split(',')]
    to_remove = [i for i in range(len(article_ranker.urls)-1) if article_ranker.sources[i] in source_list]

    article_ranker.urls = np.delete(article_ranker.urls, to_remove)
    article_ranker.sources = np.delete(article_ranker.sources, to_remove)
    article_ranker.article_embeddings = np.delete(article_ranker.article_embeddings, to_remove)
    article_ranker.source_map = {i:source for i, source in enumerate(sorted(set(article_ranker.sources)),1)}

    return ', '.join([source_display_map[s] for s in source_list])


def display_sources(source_map):
    """ Display all current sources in user-friendly format. """

    return '\n'.join(['%d. %s' % (i, source_display_map[s]) for i,s in enumerate(list(source_map.values()),1)])
