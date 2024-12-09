import os
import sys
sys.path.append('scripts/utils')
from utils_engine import search_from_web

from datetime import datetime
from tqdm import tqdm
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test_data_path', type=str, default='./datasets/test/UDK-VQA/test_raw.jsonl')
    parser.add_argument('--save_step', type=int, default=500)

    args = parser.parse_args()
    return args


def fetch_search_urls(query, freshness=''):

    engine_name = 'bing'
    exclude_list = ['www.msn.com', 'www.usatoday.com']
    query_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    urls = []
    url_infos = {}
    try:
        res_items = search_from_web(query, engine=engine_name, search_type='text', exclude_list=exclude_list, freshness=freshness)
        urls = [x['url'] for x in res_items]
        url_infos = {x['url']: x for x in res_items}

        res_items = search_from_web(query, engine=engine_name, search_type='news', exclude_list=exclude_list, freshness=freshness)
        urls2 = [x['url'] for x in res_items]
        url_infos2 = {x['url']: x for x in res_items}

        for url in urls2:
            if not url in urls:
                urls.append(url)
                url_infos[url] = url_infos2[url]

        return query_time, urls, url_infos
    except:
        return query_time, urls, url_infos




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
    print("query number = {}".format(len(query2url.keys())))

    if os.path.exists(url2info_path):
        with open(url2info_path, 'r') as f:
            url2info = json.load(f)
    else:
        url2info = {}

    save_step = args.save_step
    __iter = tqdm(query2url.keys())
    for idx, query in enumerate(__iter):
        urls = query2url[query]
        if len(urls) == 0:
            _, urls, url_infos = fetch_search_urls(query)
            query2url[query] = urls

            for url in urls:
                if not url in url2info.keys():
                    url2info[url] = url_infos[url]

        if idx % save_step == 0:
            json.dump(query2url, open(query2url_path, 'w'), indent=4)
            json.dump(url2info, open(url2info_path, 'w'), indent=4)

    json.dump(query2url, open(query2url_path, 'w'), indent=4)
    json.dump(url2info, open(url2info_path, 'w'), indent=4)

if __name__ == '__main__':
    main()