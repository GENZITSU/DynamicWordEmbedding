# -*- coding: utf-8 -*-
import time
import json
import requests
from contextlib import contextmanager


def progress_reporter(text, slack_url="https://hooks.slack.com/services/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"):
    '''該当slackチャンネルにメッセージを送信する
    '''
    data = json.dumps({"text": text, "username": 'progress_report'})
    requests.post(slack_url, data=data)
    return


@contextmanager
def timer(name, LOGGER):
    '''with文で処理にかかる時間を測る
    '''
    LOGGER.info(f"start [{name}]")
    start = time.time()
    yield
    LOGGER.info(f'[{name}] done in {time.time() - start:.0f} s')


@contextmanager
def do_job(name, LOGGER):
    '''with文で処理にかかる時間を測りながらslackに追加
    '''
    LOGGER.info(f"start [{name}]")
    start = time.time()
    try:
        yield
        LOGGER.info(f'[{name}] done in {time.time() - start:.0f} s')
        progress_reporter(f'[{name}] done in {time.time() - start:.0f} s')
    except:
        yield
        import traceback, sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        exc_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        msg = f"failed {name} \n"
        msg += ''.join('' + line for line in exc_lines)
        LOGGER.info(msg)
        progress_reporter(msg)
