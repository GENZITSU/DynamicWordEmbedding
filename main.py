import os
import glob
import pickle
import logging
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd

from core.utils import timer, do_job

# PATH
DATA_PATH = os.getenv("DATA_PATH")
PREPROCESSED_DATA_PATH = os.getenv("PREPROCESSED_DATA_PATH")
TXT_DATA_NAME = os.getenv("TXT_DATA_NAME")
print(TXT_DATA_NAME)
DW2V_PATH = os.getenv("DW2V_PATH")
PARAM_PATH = os.getenv("PARAM_PATH")

# Logger
LOGGER = logging.getLogger('JobLogging')
LOGGER.setLevel(10)
fh = logging.FileHandler('job.log')
LOGGER.addHandler(fh)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(message)s')
fh.setFormatter(formatter)
LOGGER.info("job start")

parser = argparse.ArgumentParser(description='train Dynamic Word Embeddings')
parser.add_argument('--without_preprocess', type=int, default=0, metavar='N',
                    help='if preprocessor is not neccessary, set 1')
parser.add_argument('--n_job', type=str, default="10", metavar='N',
                    help='number of cpu for multiprocessing')
parser.add_argument('--word_freq_min', type=str, default="5", metavar='N',
                    help='minmiun freqency for target word')
args = parser.parse_args()

os.environ["N_JOB"] = args.n_job
os.environ["WORD_FREQ_MIN"] = args.word_freq_min
N_JOB = int(os.getenv("N_JOB"))

if __name__ =="__main__":
    if args.without_preprocess == 0:
        # 前処理
        with do_job("preprocess tweet", LOGGER):
            from core.preprocess_tweet import preprocess_one_day_tweet

            TWEETS_PATHS = glob.glob(DATA_PATH+"alldata_20*")

            if not os.path.exists(PREPROCESSED_DATA_PATH+TXT_DATA_NAME):
                os.mkdir(PREPROCESSED_DATA_PATH+TXT_DATA_NAME)

            with Pool(processes=N_JOB) as p:
                p.map(preprocess_one_day_tweet, TWEETS_PATHS)

    # 単語の共起を確認
    with do_job("make co occ dict", LOGGER):
        from core.make_DW2V import make_unique_word2idx, make_whole_day_co_occ_dict

        TWEETS_PATHS = glob.glob(PREPROCESSED_DATA_PATH+TXT_DATA_NAME+"/*")

        # 全単語のチェック
        make_unique_word2idx(TWEETS_PATHS)

        if not os.path.exists(PREPROCESSED_DATA_PATH+"co_occ_dict_word_count/"):
            os.mkdir(PREPROCESSED_DATA_PATH+"co_occ_dict_word_count/")

        TWEETS_PATHS = glob.glob(PREPROCESSED_DATA_PATH+TXT_DATA_NAME+"/*")
        make_whole_day_co_occ_dict(TWEETS_PATHS)

    # PPMIの計算
    with do_job("make PPMI", LOGGER):
        from core.make_DW2V import make_whole_day_ppmi_list

        TWEETS_PATHS = sorted(glob.glob(PREPROCESSED_DATA_PATH+TXT_DATA_NAME+"/*"))
        DICTS_PATHS = sorted(glob.glob(PREPROCESSED_DATA_PATH+"co_occ_dict_word_count/*"))
        PATH_TUPLES = [(tweet_p, dict_p) for tweet_p, dict_p in zip(TWEETS_PATHS, DICTS_PATHS)]

        make_whole_day_ppmi_list(PATH_TUPLES)

    # DW2Vの計算
    with do_job("make DW2V", LOGGER):
        from core.make_DW2V import make_DW2V

        make_DW2V(PARAM_PATH+"params_0803.json")

