#!/usr/bin/env python3
"""
run_bot.py

Main script managing bot setup and interactions.

Input: response_manager (from response_manager.py)
Output: Functional bot (@thePoliBot on Telegram)
"""
###
import requests
import time
import argparse
import os
from requests.compat import urljoin
from response_manager import *
###

class BotHandler(object):
    """ Implements backend bot operations """

    def __init__(self, token, response_manager):
        self.token = token
        self.api_url = "https://api.telegram.org/bot{}/".format(token)
        self.response_manager = response_manager

    def get_updates(self, offset=None, timeout=30):
        params = {"timeout": timeout, "offset": offset}
        resp = requests.get(urljoin(self.api_url, "getUpdates"), params).json()
        if "result" not in resp:
            return []
        return resp["result"]

    def send_message(self, chat_id, text):
        params = {"chat_id": chat_id, "text": text}
        return requests.post(urljoin(self.api_url, "sendMessage"), params)

    def get_response(self, query):
        if query == '/start':
            return "Howdy, I'm here to provide you quick and easy access to current political media articles. Just tell me what you want to read up on!\n\nYou can also tell me any news sources you wish to have me avoid. To do so, enter the numbers corresponding to the news sources you wish to remove in the list below, separated by commas. For example, you can enter '2,4,9' to remove Chicago Tribune, Fox News, and New York Times article recommendations.\n\n" + display_sources(self.response_manager.article_ranker.source_map) + "\n\nFeel free to enter 'sources' at any time during the conversation to see your source list again."
        return self.response_manager.create_response(query)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default='')
    return parser.parse_args()


def is_unicode(text):
    return len(text) == len(text.encode())


def main():
    args = parse_args()
    token = args.token

    # Load response manager
    bot = BotHandler(token, ResponseManager())
    print('Response manager loaded.')

    print("Bot is all set to chat.")
    offset = 0
    while True:
        updates = bot.get_updates(offset=offset)
        for update in updates:
            print("Received update.")
            if "message" in update:
                chat_id = update["message"]["chat"]["id"]
                if "text" in update["message"]:
                    text = update["message"]["text"]
                    if is_unicode(text):
                        print("Update details: {}".format(update))
                        bot.send_message(chat_id, bot.get_response(update["message"]["text"]))
                    else:
                        bot.send_message(chat_id, "Mind cleaning up your text a bit? Strange characters confuse me. :(")
            offset = max(offset, update['update_id'] + 1)
        time.sleep(1)

if __name__ == "__main__":
    main()
