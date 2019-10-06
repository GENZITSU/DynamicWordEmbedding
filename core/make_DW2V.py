import os
import json
import glob
import pickle
import logging
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd

from core.utils import timer, do_job
from core import util_timeCD as util


MeCab_DICT_PATH = os.getenv("MeCab_DICT_PATH", "/usr/lib/mecab/dic/mecab-ipadic-neologd/")
PREPROCESSED_DATA_PATH = os.getenv("PREPROCESSED_DATA_PATH",
                                    "/mnt/NAS0CAC8A/k-syo/DW2V/preprocessed_data/")
N_JOB = int(os.getenv("NJOB", "10"))
WORD_FREQ_MIN = int(os.getenv("WORD_FREQ_MIN", "5"))
DW2V_PATH = os.getenv("DW2V_PATH", "/mnt/NAS0CAC8A/k-syo/DW2V/")

# Logging
LOGGER = logging.getLogger('JobLogging')
LOGGER.setLevel(10)
fh = logging.FileHandler('preprocess.log')
LOGGER.addHandler(fh)
formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(message)s')
fh.setFormatter(formatter)
LOGGER.info("logging start")

def check_one_day_word(tweet_path):
    '''1文書中に含まれる単語のsetを保存する
    tweet_path:  str
    tweet(文書)のDataFrameが保存されているPath
    DataFrameには文書が入っているカラム"body"が必要
    NOTE
    ----
    tweet_path[-17:-7]中にある日付を保存する際の名前にしている
    日付の位置が違う場合は適宜変更のこと
    '''
    date = tweet_path[-17:-7]
    with timer(f"check one day word {date}", LOGGER):
        with open(tweet_path, mode="rb") as f:
            df = pickle.load(f)
    
        tweets = df["body"].values
        del df #不要な変数を削除
        tweets = " ".join(tweets)
        tweets = tweets.split(" ")
    
        word_set = set(tweets)
        
        with open(PREPROCESSED_DATA_PATH+"word_sets/"+date+".pickle", mode="wb") as f:
            pickle.dump(word_set, f)
    return


def make_unique_word2idx(TWEETS_PATHS):
    '''単語: idのdictionaryを作る
    TWEETS_PATHS: [str, str, ...]
    tweet(文書)のDataFrameが保存されているPathのリスト
    DataFrameには文書が入っているカラム"body"が必要
    '''
    if not os.path.exists(PREPROCESSED_DATA_PATH+"word_sets"):
            os.mkdir(PREPROCESSED_DATA_PATH+"word_sets")
    with Pool(processes=N_JOB) as p:
        p.map(check_one_day_word, TWEETS_PATHS)
    
    # 全単語のsetを結合
    WORD_SET_PATHS = glob.glob(PREPROCESSED_DATA_PATH+"word_sets/*")
    unique_word_set = set()
    for word_set_path in WORD_SET_PATHS:
        with open(word_set_path, mode="rb") as f:
            word_set = pickle.load(f)
        unique_word_set = unique_word_set.union(word_set)

    # dictionaryを作成
    word2idx = {w:i for i,w in enumerate(unique_word_set)}
    
    # 保存
    with open(PREPROCESSED_DATA_PATH+"unique_word2idx.pickle", mode="wb") as f:
        pickle.dump(word2idx, f)
    return


def make_co_occ_dict(tweet_path, window_size=7):
    '''単語の共起と単語の出現回数を保存する
    Params
    ------
    tweet_path: str
    tweet(文書)のDataFrameが保存されているPath
    DataFrameには文書が入っているカラム"body"が必要
    window size: int
    共起をカウントする検索幅、自身+前後の単語の総数
    奇数推奨
    Return
    ------
    None
    (単語の共起の隣接リスト, 単語の出現回数 array)のtuple
    を保存する
    NOTE
    ----
    tweet_path[-17:-7]中にある日付を保存する際の名前にしている
    日付の位置が違う場合は適宜変更のこと
    '''
    date = tweet_path[-17:-7]
    with open(PREPROCESSED_DATA_PATH+"unique_word2idx.pickle", mode="rb") as f:
        word2idx = pickle.load(f)
    unique_word_num = len(word2idx.keys())

    with timer(f"load {date} data", LOGGER):
        with open(tweet_path, mode="rb") as f:
            df = pickle.load(f)
        
        tweets = df["body"].values
        np.random.shuffle(tweets)
        del df #不要な変数を削除
        tweets = " ".join(tweets)
        splited_tweet = tweets.split(" ")
        del tweets #不要な変数を削除
    
    # 単語の共起を記録
    # with timer(f"make co_occ_dict {date}", LOGGER):
    with do_job(f"make co_occ_dict {date}", LOGGER):
        co_occ_dict = {w: [] for w in word2idx.keys()}
        word_count = np.zeros(unique_word_num)
        tweet_sum_len = len(splited_tweet)
        for i, w in enumerate(splited_tweet):
            try:
                word_count[word2idx[w]] += 1
                for window_idx in range(1, int((window_size + 1)/2+1)):
                    if i - window_idx >= 0:
                        co_list = co_occ_dict[w]
                        co_list.append(splited_tweet[i-window_idx])
                        co_occ_dict[w] = co_list

                    if i+window_idx < tweet_sum_len:
                        co_list = co_occ_dict[w]
                        co_list.append(splited_tweet[i+window_idx])
                        co_occ_dict[w] = co_list
            except:
                continue
                
        del co_list #不要な変数を削除
        del splited_tweet #不要な変数を削除
        
        # 保存
        save_path = PREPROCESSED_DATA_PATH+"co_occ_dict_word_count/"+date+".pickle"
        with open(save_path, mode="wb") as f:
            pickle.dump((co_occ_dict, word_count), f)
            
    return


def make_one_day_ppmi_list(path_tuple):
    '''各時刻でのPPMIを計算する
    based on eq (1) from
    https://arxiv.org/pdf/1703.00607.pdf
    Prams:
    ------
    path_tuple: (tweet_path, co_occ_path)
        tweet_path:  str
        tweet(文書)のDataFrameが保存されているPath
        DataFrameには文書が入っているカラム"body"が必要
        co_occ_path:  str
        単語の共起連結リストと単語の出現頻度 arrayのtuple
        が保存されているpath
    Return:
    ------
    None
    PPMIのarrayを保存する
    NOTE
    ----
    - tweet_path[-17:-7]中にある日付を保存する際の名前にしている
      日付の位置が違う場合は適宜変更のこと
    - WORD_FREQ_MINによって対象単語が制限される
    '''
    tweet_path, co_occ_path = path_tuple
    date = tweet_path[-17:-7]

    with open(co_occ_path, mode="rb") as f:
        co_occ_dict, word_count = pickle.load(f)

    # 単語とidxのマッピングをロード
    with open(PREPROCESSED_DATA_PATH+"unique_word2idx.pickle", mode="rb") as f:
        word2idx = pickle.load(f)
    with open(PREPROCESSED_DATA_PATH+"filtered_word_idx_converter.pickle", mode="rb") as f:
        idxconverter = pickle.load(f)
    idx2word = {i: w for w, i in word2idx.items()}

    # with timer(f"calc ppmi_list {date}", LOGGER):
    with do_job(f"calc ppmi_list {date}", LOGGER):
        # |D| : total number of tokens in corpus
        D = get_number_of_tokens(tweet_path)

        ppmi_list = []
        for target_word_idx in idxconverter.keys():
            target_word = idx2word[target_word_idx]
            cnt = Counter(co_occ_dict[target_word])
            for co_occ_word, co_occ_freq in cnt.most_common():
                co_occ_word_idx = word2idx[co_occ_word]
                if idxconverter.get(co_occ_word_idx):
                    # 出現頻度の低い単語を無視
                    ppmi = calc_ppmi(co_occ_freq, word_count, target_word_idx, co_occ_word_idx, D)
                    if ppmi > 0:
                        # sparse matrixを作成するため0以上のみ保存
                        target_word_idx_ = idxconverter[target_word_idx]
                        co_occ_word_idx_ = idxconverter[co_occ_word_idx]
                        ppmi_list.append([ppmi, target_word_idx_, co_occ_word_idx_])

        with open(PREPROCESSED_DATA_PATH+"ppmi_list/"+date+".pickle", mode="wb") as f:
            pickle.dump(np.array(ppmi_list), f)

    return


def make_whole_day_ppmi_list(PATH_TUPLES):
    '''全時刻のPPMIを計算する
    based on eq (1) from
    https://arxiv.org/pdf/1703.00607.pdf
    Prams:
    ------
    path_tuple: (tweet_path, co_occ_path)
        tweet_path:  str
        tweet(文書)のDataFrameが保存されているPath
        DataFrameには文書が入っているカラム"body"が必要
        co_occ_path:  str
        単語の共起連結リストと単語の出現頻度 arrayのtuple
        が保存されているpath
    Return:
    ------
    None
    PPMIのarrayを保存する
    NOTE
    ----
    - tweet_path[-17:-7]中にある日付を保存する際の名前にしている
      日付の位置が違う場合は適宜変更のこと
    - WORD_FREQ_MINによって対象単語が制限される
    '''
    ### 対象単語の制限
    with timer("filter words", LOGGER):
        with open(PREPROCESSED_DATA_PATH+"unique_word2idx.pickle", mode="rb") as f:
            word2idx = pickle.load(f)

        # 最新のデータ中の出現回数でフィルタリング
        DICT_PATHS = sorted(glob.glob(PREPROCESSED_DATA_PATH+"co_occ_dict_word_count/*"))
        with open(DICT_PATHS[-1], mode="rb") as f:
            _, word_count = pickle.load(f)
            
        # word -> idxのマッピングの制限で実現する
        idxconverter = {}
        new_idx = 0
        for old_idx in word2idx.values():
            if word_count[old_idx] >= WORD_FREQ_MIN :
                idxconverter[old_idx] = new_idx
                new_idx += 1
        
        # 保存
        with open(PREPROCESSED_DATA_PATH+"filtered_word_idx_converter.pickle", mode="wb") as f:
            pickle.dump(idxconverter, f)
        
        del word2idx, word_count, idxconverter

    if not os.path.exists(PREPROCESSED_DATA_PATH+"ppmi_list/"):
            os.mkdir(PREPROCESSED_DATA_PATH+"ppmi_list/")

    with Pool(processes=N_JOB) as p:
        p.map(make_one_day_ppmi_list, PATH_TUPLES)

    return


def get_number_of_tokens(tweet_path):
    '''get |D| : total number of tokens in corpus
    based on eq (1) from
    https://arxiv.org/pdf/1703.00607.pdf

    tweet_path:  str
    tweet(文書)のDataFrameが保存されているPath
    DataFrameには文書が入っているカラム"body"が必要
    NOTE
    ----
    tweet_path[-17:-7]中にある日付を保存する際の名前にしている
    日付の位置が違う場合は適宜変更のこと
    '''
    with open(tweet_path, mode="rb") as f:
        tweet_df = pickle.load(f)
    tweets = tweet_df["body"].values
    del tweet_df
    tweets = " ".join(tweets)
    D = len(tweets)
    return D


def calc_ppmi(co_occ_freq, word_count, target_word_idx, co_occ_word_idx, D):
    '''単語舞のPPMIを計算する
    based on eq (1) from
    https://arxiv.org/pdf/1703.00607.pdf
    '''
    pmi = np.log(co_occ_freq) + np.log(D)
    pmi -= np.log(word_count[target_word_idx])
    pmi -= np.log(word_count[co_occ_word_idx])
    return max(pmi, 0)


def make_DW2V(param_path, EPS=1e-4):
    '''時系列毎の単語ベクトルを作成する
    based on eq (8) from
    https://arxiv.org/pdf/1703.00607.pdf
    param_path: str
    ハイパラを記したjsonファイルのpath
    '''

    # PPMIのパスを読み込む
    PPMI_PATHS = sorted(glob.glob(PREPROCESSED_DATA_PATH+"ppmi_list/*"))
    # number of time spans
    T = len(PPMI_PATHS)

    ## PARAMETERS
    params = json.load(open(param_path, mode="r"))
    ITERS = params["ITERS"]
    lam = params["lam"] # weight decay
    gam = params["gam"] # forcing regularizer
    tau = params["tau"]  # smoothing regularizer
    embed_size  = params["embed_size"]
    emph = params["emph"] # emphasize the nonzero


    # 保存先の確保
    savefile = DW2V_PATH+"Lam_"+str(lam)+"_Tau_"+str(tau)+"_Gam_"+str(gam)+"_A_"+str(emph)+"/"
    if not os.path.exists(savefile):
            os.mkdir(savefile)

    # ベクトルの初期化
    with open(PREPROCESSED_DATA_PATH+"filtered_word_idx_converter.pickle", mode="rb") as f:
        idxconverter = pickle.load(f)
    vocab_size = len(idxconverter.keys())
    del idxconverter
    batch_size = vocab_size
    b_ind= util.getbatches(vocab_size, batch_size)
    Ulist,Vlist = util.initvars(vocab_size, T, embed_size)

    # 学習開始
    diffs = []
    for iteration in range(ITERS):
        with do_job(f"iter {iteration+1} / {ITERS}", LOGGER):
            try:
                Ulist = pickle.load(open(f"{savefile}ngU_iter{iteration}.pickle", mode="rb" ))
                Vlist = pickle.load(open(f"{savefile}ngV_iter{iteration}.pickle", mode="rb" ))
            except(IOError):
                pass

            # shuffle times
            if iteration == 0:
                times = np.arange(T)
            else:
                times = np.random.permutation(range(T))

            for t in times:
                with timer(f"Update {t} th span Vector", LOGGER):
                    f = PPMI_PATHS[t]
                    pmi = util.getmat(f,vocab_size,False)
                    for ind in b_ind:
                        ## UPDATE U, V
                        pmi_seg = pmi[:,ind]

                        if t==0:
                            vp = np.zeros((len(ind),embed_size))
                            up = np.zeros((len(ind),embed_size))
                            iflag = 1
                        else:
                            vp = Vlist[t-1][ind]
                            up = Ulist[t-1][ind]
                            iflag = 0

                        if t==T-1:
                            vn = np.zeros((len(ind),embed_size))
                            un = np.zeros((len(ind),embed_size))
                            iflag = 1
                        else:
                            vn = Vlist[t+1][ind]
                            un = Ulist[t+1][ind]
                            iflag = 0
                        Vlist[t][ind] = util.update(Ulist[t],emph*pmi_seg,vp,vn,
                                                    lam,tau,gam,ind,embed_size,iflag)
                        Ulist[t][ind] = util.update(Vlist[t],emph*pmi_seg,up,un,
                                                    lam,tau,gam,ind,embed_size,iflag)

            # 保存
            pickle.dump(Ulist, open(f"{savefile}ngU_iter{iteration}.pickle", mode="wb"))
            pickle.dump(Vlist, open(f"{savefile}ngV_iter{iteration}.pickle", mode="wb"))

            if iteration >= 2:
                # HDDの節約
                os.remove(f"{savefile}ngU_iter{iteration-2}.pickle")
                os.remove(f"{savefile}ngV_iter{iteration-2}.pickle")

            diff_U, diff_V, diff_U_V = util.check_diff(iteration, savefile)
            diffs.append([diff_U, diff_V, diff_U_V])

            # ほとんど変化しなくなったら終了
            if (diff_U + diff_V)/2 < EPS and diff_U != 0.:
                break
                
    return diffs