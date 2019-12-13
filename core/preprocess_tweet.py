# -*- coding: utf-8 -*-
'''
Twitter Dataに対する 前処理関数達
'''
import os
import re
import ast
import pickle
import logging

import MeCab
import pandas
import neologdn
import pandas as pd

from core.utils import timer

MeCab_DICT_PATH = os.getenv("MeCab_DICT_PATH")
PREPROCESSED_DATA_PATH = os.getenv("PREPROCESSED_DATA_PATH")
TXT_DATA_NAME = os.getenv("TXT_DATA_NAME")

# Logging
LOGGER = logging.getLogger('JobLogging')
LOGGER.setLevel(10)
fh = logging.FileHandler('preprocess.log')
LOGGER.addHandler(fh)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(message)s')
fh.setFormatter(formatter)
LOGGER.info("logging start")

def preprocess_tweet(text):
    '''jsonファイルのnull, true, Falseを修正する
    text : str
    '''
    text = text.replace("null", "None")
    text = text.replace(r"\N", "\n")
    text = text.replace("true", "True")
    text = text.replace("false", "False")
    return text


def remove_noise(string):
    '''ツイートに対する前処理
    string: str
    '''
    # リプライの@を削除
    string = re.sub(r"@[A-Z0-9a-z,_/:。]+", "@USER", string)
    # リンク掲載を削除
    string = re.sub(r"https:[A-Z0-9a-z,./]+", "URL", string)
    # リンク掲載を削除
    string = re.sub(r"http:[A-Z0-9a-z,./]+", "URL", string)
    # \nを削除
    string = re.sub(r"\n", "", string)
    # \rを削除
    string = re.sub(r"\r", "", string)
    # &ampを&に
    string = re.sub(r"&amp", "&", string)
    # &gt;を>に
    string = re.sub(r"&gt;", ">", string)
    # &lt;を<に
    string = re.sub(r"&lt;", "<", string)

    return string


def preprocess_one_day_tweet(tweet_path):
    '''pathで指定されるjsonファイルを読んでツイートのDataFrameを作成する
    tweet_path:  str
    NOTE
    ----
    tweet_path[-15:-5]中にある日付を保存する際の名前にしている
    日付の位置が違う場合は適宜変更のこと
    '''
    date = tweet_path[-15:-5]
    ### Twieetの読み込み
    with timer(f"reading {date}", LOGGER):
        tweet_list = []
        with open(tweet_path) as f:
            l = f.readline()
            while l:
                l = f.readline()
                try:
                    if len(l) == 0:
                        break
                    tweet = preprocess_tweet(l)
                    tweet_dict = ast.literal_eval(tweet)
                    tweet_dict["body"] = tweet_dict["body"].encode('utf-16',
                                                            'surrogatepass').decode('utf-16')
                    tweet_list.append(tweet_dict)
                except:
                    continue

        tweet_df = pd.DataFrame(tweet_list)
        tweet_df = tweet_df[["body", "created_at"]]
        tweet_list = None # 不必要な変数を削除


    ### Tweet分の前処理
    # ノイズを処理
    with timer(f"cleaning {date}", LOGGER):
        # tweet_df["source_url"] = tweet_df["source_url"]\
        #                             .map(lambda x: x.encode('utf-16','surrogatepass').decode('utf-16'))
        tweet_df["body"] = tweet_df["body"].map(lambda x: remove_noise(x))

    # 標準化
    with timer(f"normalizing {date}", LOGGER):
        tweet_df["body"] = tweet_df["body"].map(lambda x: neologdn.normalize(x, repeat=3))

    # 分かち書き
    with timer(f"tokenizing {date}", LOGGER):
        m = MeCab.Tagger(f"-Owakati -d {MeCab_DICT_PATH}")
        tweet_df["body"] = tweet_df["body"].map(lambda x: m.parse(x))

    # 保存
    with open(PREPROCESSED_DATA_PATH+TXT_DATA_NAME+"/"+date+".pickle", mode="wb") as f:
        pickle.dump(tweet_df, f)

    return None
