import time
from selenium import webdriver
import chromedriver_binary
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# def get_field_text_if_exists(item, selector):
#     """Extracts a field by a CSS selector if exists."""
#     try:
#         return item.find_element_by_css_selector(selector).text
#     except NoSuchElementException:
#         return ""
#
#
# def get_link_if_exists(item, selector):
#     """Extracts an href attribute value by a CSS selector if exists."""
#     try:
#         return item.find_element_by_css_selector(selector).get_attribute("href")
#     except NoSuchElementException:
#         return ""


driver = webdriver.Chrome()
wait = WebDriverWait(driver, 10)

#driver.get("https://www.zebra.com/us/en/partners/partner-application-locator.html")
driver.get("http://www.foxnews.com/politics.html")

# location = driver.find_element_by_css_selector('.partnerLocation input')
# location.clear()
# location.send_keys("Colorado, USA")

# select the first suggestion from a suggestion dropdown
#dropdown_suggestion = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'ul[id^=typeahead] li a')))
#dropdown_suggestion.click()

# click more until no more results to load
for i in range(100):
    try:
        #more_button = wait.until(EC.visibility_of_element_located((By.CLASS_NAME, 'showmore-bg'))).click()
        more_button = wait.until(EC.visibility_of_element_located((By.CLASS_NAME, 'load-more'))).click()
        #more_button = driver.find_element_by_class_name('load-more')
        #driver.execute_script('arguments[0].click();', more_button)
        print('clicked')
        time.sleep(1)
    except TimeoutException:
        break

# wait for results to load
#wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, '.seclection-result .partners-detail')))
#wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, '.story-12')))
time.sleep(5)
# parse results
#for result in driver.find_elements_by_css_selector('.seclection-result .partners-detail'):
all_urls = []
for result in driver.find_elements_by_xpath('//a[@href]'):
    article_url = result.get_attribute('href')
    # name = get_field_text_if_exists(result, 'a')
    # address = get_field_text_if_exists(result, '.fullDetail-cmpAdres')
    # phone = get_field_text_if_exists(result, '.fullDetail-cmpAdres p[ng-if*=phone]')
    # website = get_link_if_exists(result, 'a[ng-if*=website]')

    #print(article_url)
    all_urls.append(article_url)

driver.quit()
print(all_urls)
