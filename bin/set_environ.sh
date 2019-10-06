#!/usr/bin/env bash

### Set up training environment.
# source data path
export DATA_PATH = "<your data_path>"
# mecab dictionary path
export MeCab_DICT_PATH = "<your Mecab dict dict>"
# mid output directory
export PREPROCESSED_DATA_PATH = "/home/jovyan/DW2V/preprocessed_data/"
# final output directory
export DW2V_PATH = "/home/jovyan/DW2V/"
# param path for DW2V
export PARAM_PATH = "/home/jovyan/DW2V/DynamicWordEmbedding/params/DW2V/"
# api for slack notification
export SLACK_URL = "<your slack web hock url>"

export NJOB = "10"
export WORD_FREQ_MIN = "5"