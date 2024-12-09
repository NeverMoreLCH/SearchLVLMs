import os
import sys
sys.path.append('scripts/utils')
from utils import check_fetched_text, text_chunk, fetch_by_newspaper

import json
from tqdm import tqdm
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test_data_path', type=str, default='./datasets/test/UDK-VQA/test_raw.jsonl')
    parser.add_argument('--save_step', type=int, default=500)

    args = parser.parse_args()
    return args


def fetch_web_info_by_crawl(url):
    try:
        fetched_dict = fetch_by_newspaper(url)
    except:
        fetched_dict = {'fetched_title': '', 'fetched_text': ''}

    fetched_text = fetched_dict['fetched_text']
    fetched_text_chunked_list = []
    if check_fetched_text(fetched_text):
        fetched_text_chunked_list = text_chunk(fetched_text, 150)

    return fetched_text, fetched_text_chunked_list


def main():

    args = parse_args()

    test_data_path = args.test_data_path

    dataset_name = os.path.dirname(test_data_path).split('/')[-1]
    query2url_path = './intermediate_files/{}/query2url.json'.format(dataset_name)
    url2info_path = './intermediate_files/{}/url2info.json'.format(dataset_name)

    folder_path = os.path.dirname(query2url_path)
    os.makedirs(folder_path, exist_ok=True)

    if os.path.exists(query2url_path):
        with open(query2url_path, 'r') as f:
            query2url = json.load(f)
    else:
        query2url = {}

    url2info = {}
    for k, v in query2url.items():
        for url in v:
            url2info[url] = {}
    url2info = dict(sorted(url2info.items()))

    candidate = []
    for k, v in url2info.items():
        candidate.append(k)

    total_sample_num = len(candidate)
    print(len(url2info.keys()), total_sample_num)

    last_url_type = ''
    valid_num = 0
    save_step = args.save_step
    __iter = tqdm(candidate)
    for idx, url in enumerate(__iter):
        now_url_type = url[:20]
        if now_url_type == last_url_type:
            time.sleep(2)

        infos = url2info[url]
        do_search = False
        if not "have_searched" in infos.keys():
            do_search = True

        if do_search:
            fetched_text, fetched_text_chunked_list = fetch_web_info_by_crawl(url)
            url2info[url]['fetched_text'] = fetched_text
            url2info[url]['fetched_text_chunked_list'] = fetched_text_chunked_list
            url2info[url]['have_searched'] = True

        if len(url2info[url]['fetched_text_chunked_list']) > 0:
            valid_num = valid_num + 1
        __iter.set_description("valid num = {}/{}".format(valid_num, idx+1))

        last_url_type = url[:20]
        if idx % save_step == 0:
            json.dump(url2info, open(url2info_path, 'w'), indent=4)

    json.dump(url2info, open(url2info_path, 'w'), indent=4)

if __name__ == '__main__':
    main()