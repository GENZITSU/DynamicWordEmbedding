import os
import glob
import pickle
import logging
from multiprocessing import Pool

import numpy as np
import pandas as pd

from core.utils import timer, do_job

# PATH
DATA_PATH = os.getenv("DATA_PATH", "/mnt/NAS0CAC8A/collaborations/dentsuPR2019/")
PREPROCESSED_DATA_PATH = os.getenv("PREPROCESSED_DATA_PATH",
                                    "/mnt/NAS0CAC8A/k-syo/DW2V/preprocessed_data/")
N_JOB = int(os.getenv("NJOB", "10"))
DW2V_PATH = os.getenv("DW2V_PATH", "/mnt/NAS0CAC8A/k-syo/DW2V/")
PARAM_PATH = os.getenv("PARAM_PATH", "/home/k-syo/DynamicWordEmbedding/params/DW2V/")

# Logger
LOGGER = logging.getLogger('JobLogging')
LOGGER.setLevel(10)
fh = logging.FileHandler('job.log')
LOGGER.addHandler(fh)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(message)s')
fh.setFormatter(formatter)
LOGGER.info("job start")

if __name__ =="__main__":
    # 前処理
    with do_job("preprocess tweet", LOGGER):
        from core.preprocess_tweet import preprocess_one_day_tweet

        TWEETS_PATHS = glob.glob(DATA_PATH+"alldata_20*")

        if not os.path.exists(PREPROCESSED_DATA_PATH+"tokenized_tweets"):
            os.mkdir(PREPROCESSED_DATA_PATH+"tokenized_tweets")

        with Pool(processes=N_JOB) as p:
            p.map(preprocess_one_day_tweet, TWEETS_PATHS)

    # 単語の共起を確認
    with do_job("make co occ dict", LOGGER):
        from core.make_DW2V import make_unique_word2idx, make_co_occ_dict

        TWEETS_PATHS = glob.glob(PREPROCESSED_DATA_PATH+"tokenized_tweets/*")

        # 全単語のチェック
        make_unique_word2idx(TWEETS_PATHS)

        if not os.path.exists(PREPROCESSED_DATA_PATH+"co_occ_dict_word_count/"):
            os.mkdir(PREPROCESSED_DATA_PATH+"co_occ_dict_word_count/")

        TWEETS_PATHS = glob.glob(PREPROCESSED_DATA_PATH+"tokenized_tweets/*")
        with Pool(processes=N_JOB) as p:
            p.map(make_co_occ_dict, TWEETS_PATHS)

    # PPMIの計算
    with do_job("make PPMI", LOGGER):
        from core.make_DW2V import make_whole_day_ppmi_list

        TWEETS_PATHS = sorted(glob.glob(PREPROCESSED_DATA_PATH+"tokenized_tweets/*"))
        DICTS_PATHS = sorted(glob.glob(PREPROCESSED_DATA_PATH+"co_occ_dict_word_count/*"))
        PATH_TUPLES = [(tweet_p, dict_p) for tweet_p, dict_p in zip(TWEETS_PATHS, DICTS_PATHS)]

        make_whole_day_ppmi_list(PATH_TUPLES)

    # DW2Vの計算
    with do_job("make DW2V", LOGGER):
        from core.make_DW2V import make_DW2V

        make_DW2V(PARAM_PATH+"params_0803.json")

