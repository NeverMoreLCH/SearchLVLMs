import os
import sys
sys.path.append(os.environ.get("llama3_dir"))
# settings for llama3
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12339'
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"


sys.path.append('scripts/utils')
from utils import parse_ner_res, gpt_text_only
from utils_engine import search_from_web


from typing import List, Optional
from llama import Dialog, Llama
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from sklearn.cluster import KMeans

import torch
import clip
import json
import time
import numpy as np
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test_data_path', type=str, default='./datasets/test/UDK-VQA/test_raw.jsonl')
    parser.add_argument('--test_img_dir', type=str, default='./datasets/test/UDK-VQA/images')

    parser.add_argument('--ner_model_path', type=str, default='/cpfs01/user/lichuanhao/huggingface_cache/hub/models--dslim--bert-large-NER/snapshots/13e784dccceca07aee7a7aab4ad487c605975423')
    parser.add_argument('--llama3_ckpt_dir', type=str, default='/cpfs01/user/lichuanhao/huggingface_cache/llama3/Meta-Llama-3-8B-Instruct/')
    parser.add_argument('--llama3_tokenizer_path', type=str, default='/cpfs01/user/lichuanhao/huggingface_cache/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model')
    parser.add_argument('--clip_dir', type=str, default='/cpfs01/user/lichuanhao/huggingface_cache/clip')
    
    parser.add_argument('--save_step', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()
    return args



def _cluster(model, processor, device, texts, cluster_num):
    cluster_num = min(cluster_num, len(texts))
    if cluster_num == 0:
        return []

    ret_list = []
    text_features = []
    
    model = model.to(device)
    model = model.float()
    
    for text in texts:
        inputs = clip.tokenize(text).to(device) 
        # print(f"Inputs device: {inputs.device}, Model device: {device}")
        with torch.no_grad():
            features = model.encode_text(inputs)
        text_features.append(features.cpu().numpy())

    text_features = np.concatenate(text_features, axis=0)

    n_clusters = cluster_num
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(text_features)

    clustered_texts = {}
    for i, label in enumerate(kmeans.labels_):
        if label not in clustered_texts:
            clustered_texts[label] = []
        clustered_texts[label].append((texts[i], np.linalg.norm(text_features[i] - kmeans.cluster_centers_[label])))

    for i, texts_distances in clustered_texts.items():
        closest_text, closest_distance = min(texts_distances, key=lambda x: x[1])
        ret_list.append(closest_text)

    ret_list.append(', '.join(ret_list))

    return ret_list

def gen_query_by_gpt(question):
    temperature = 0.3
    pair_num = 1
    prompt = "'Question: {}\n\nDo not try to answer the question, just print the most informative no more than three entities in the question. Put them on one line and separate them with comm."
    gen_lvlm = "gpt-3.5-turbo"

    text = prompt.format(question)
    res_list = gpt_text_only(text, temperature=temperature, pair_num=pair_num, model=gen_lvlm)
    # print('res_list = {}'.format(res_list))

    return res_list[0].split(', ')

def gen_query_by_llama3(question, generator):

    llama3_prompt = "Question: {}\n\nDo not try to answer the question, just print the most informative no more than three entities in the question. Put them on one line and separate them with comm."
    text = llama3_prompt.format(question)
    dialogs: List[Dialog] = [
        [{"role": "user", "content": text}],
    ]
    results = generator.chat_completion(
        dialogs,
        max_gen_len=512,
        temperature=0.6,
        top_p=0.9,
    )

    res = [x.strip() for x in results[0]["generation"]["content"].split(',')]
    if "" in res:
        res.remove("")
    return res


def extract_q_query(question, nlp, generator):

    queries = set()
    now_time = 0
    while(True):
        now_time = now_time + 1
        try:
            ner_set = set([x['entity'].lower() for x in parse_ner_res(question, nlp(question))])
            gpt_set = set([x.lower() for x in gen_query_by_gpt(question)])
            lma_set = set([x.lower() for x in gen_query_by_llama3(question, generator)])

            queries = queries | ner_set
            queries = queries | gpt_set
            queries = queries | lma_set
            break
            
        except:
            if now_time >= 3:
                break
            time.sleep(1)
    return queries

def extract_v_query(img_path, nlp):

    now_time = 0
    queries = set()
    while True:
        now_time = now_time + 1
        try:
            res = search_from_web(img_path, engine='bing', search_type='visual', nlp=nlp)
            queries = set([res['search_str']])
            break
        except:
            if now_time >= 3:
                break
            time.sleep(1)
    
    return queries




def main():

    args = parse_args()

    test_data_path = args.test_data_path
    test_img_dir = args.test_img_dir
    test_data = [json.loads(q) for q in open(os.path.expanduser(test_data_path), "r")]

    # load NER, llama3
    ner_model_path = args.ner_model_path
    llama3_ckpt_dir = args.llama3_ckpt_dir
    llama3_tokenizer_path = args.llama3_tokenizer_path

    ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
    ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_path)
    nlp = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)
    llama3_generator = Llama.build(
        ckpt_dir=llama3_ckpt_dir,
        tokenizer_path=llama3_tokenizer_path,
        max_seq_len=512,
        max_batch_size=6,
    )

    device = torch.device(args.device)
    clip_model, clip_processor = clip.load("ViT-B/32", download_root=args.clip_dir, device=device)

    dataset_name = os.path.dirname(test_data_path).split('/')[-1]
    id2query_path = './intermediate_files/{}/id2query.json'.format(dataset_name)
    id2clusterquery_path = './intermediate_files/{}/id2clusterquery.json'.format(dataset_name)
    id2allquery_path = './intermediate_files/{}/id2allquery.json'.format(dataset_name)
    query2url_path = './intermediate_files/{}/query2url.json'.format(dataset_name)

    folder_path = os.path.dirname(id2query_path)
    os.makedirs(folder_path, exist_ok=True)

    if os.path.exists(id2clusterquery_path):
        with open(id2clusterquery_path, 'r') as f:
            id2clusterquery = json.load(f)
    else:
        id2clusterquery = {}

    if os.path.exists(id2allquery_path):
        with open(id2allquery_path, 'r') as f:
            id2allquery = json.load(f)
    else:
        id2allquery = {}

    if os.path.exists(id2query_path):
        with open(id2query_path, 'r') as f:
            id2query = json.load(f)
    else:
        id2query = {}

    if os.path.exists(query2url_path):
        with open(query2url_path, 'r') as f:
            query2url = json.load(f)
    else:
        query2url = {}

    save_step = args.save_step
    __iter = tqdm(test_data)
    for idx, sample in enumerate(__iter):
        __iter.set_description('processed query num = {}'.format(len(query2url.keys())))

        question_id = sample["question_id"]
        question = sample["text"].split('Question: ')[-1].split('\nAnswers:')[0]
        image_name = sample["image"]
        img_path = os.path.join(test_img_dir, image_name)

        if not question_id in id2query.keys():
            q_queries = extract_q_query(question, nlp, llama3_generator)
        else:
            q_queries = set(id2query[question_id])
            if len(q_queries) == 0:
                q_queries = extract_q_query(question, nlp, llama3_generator)

        if not image_name in id2query.keys():
            v_queries = extract_v_query(img_path, nlp)
        else:
            v_queries = set(id2query[image_name])

        all_queries = q_queries | v_queries
        tmp_queries = []
        for x in all_queries:
            if len(x) > 5:
                tmp_queries.append(x)
        clustered_queries = set(_cluster(clip_model, clip_processor, device, tmp_queries, 3))
        all_queries = all_queries | clustered_queries

        id2query[question_id] = list(q_queries)
        id2query[image_name] = list(v_queries)
        id2clusterquery[question_id] = list(clustered_queries)
        id2allquery[question_id] = list(all_queries)

        for query in clustered_queries:
            if not query in query2url.keys():
                query2url[query] = []

        if idx % save_step == 0:
            json.dump(id2query, open(id2query_path, 'w'), indent=4)
            json.dump(id2allquery, open(id2allquery_path, 'w'), indent=4)
            json.dump(id2clusterquery, open(id2clusterquery_path, 'w'), indent=4)
            json.dump(query2url, open(query2url_path, 'w'), indent=4)

    json.dump(id2query, open(id2query_path, 'w'), indent=4)
    json.dump(id2allquery, open(id2allquery_path, 'w'), indent=4)
    json.dump(id2clusterquery, open(id2clusterquery_path, 'w'), indent=4)
    json.dump(query2url, open(query2url_path, 'w'), indent=4)


if __name__ == '__main__':
    main()