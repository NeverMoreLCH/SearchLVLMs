import json
import os
from tqdm import tqdm

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test_data_path', type=str, default='./datasets/test/UDK-VQA/test_raw.jsonl')
    parser.add_argument('--prompt', type=str, default="How helpful is this context in answering the question based on the image? Choose the best option.\n\nContext: {}\nQuestion: {}\nOptions:\nA. 1.0\nB. 0.8\nC. 0.6\nD. 0.4\nE. 0.2\nF. 0.0\n")

    args = parser.parse_args()
    return args

def get_id2infos(test_data_path):

    id2infos = {}
    test_list = [json.loads(q) for q in open(os.path.expanduser(test_data_path), "r")]
    __iter = tqdm(test_list)
    for idx, sample in enumerate(__iter):
        question = sample["text"].split("Question: ")[-1].split("\nAnswers:")[0]
        question_id = sample["question_id"]
        image_name = sample["image"]

        choices_str = sample["text"].split("\nAnswers:\n")[-1].split("\n\nAnswer")[0]
        ch1 = choices_str.split("A. ")[-1].split('\n')[0]
        ch2 = choices_str.split("B. ")[-1].split('\n')[0]
        ch3 = choices_str.split("C. ")[-1].split('\n')[0]
        ch4 = choices_str.split("D. ")[-1].split('\n')[0]

        id2infos[question_id] = {
            "question": question,
            "image": image_name,
            "choices": [ch1, ch2, ch3, ch4],
            "gt_choice": sample["category"],
            "gt_ans": sample["answer_text"]
        }

    return id2infos

def main():

    args = parse_args()

    prompt = args.prompt
    test_data_path = args.test_data_path

    dataset_name = os.path.dirname(test_data_path).split('/')[-1]
    id2allquery_path = './intermediate_files/{}/id2allquery.json'.format(dataset_name)
    query2url_path = './intermediate_files/{}/query2url.json'.format(dataset_name)
    url2info_path = './intermediate_files/{}/url2info.json'.format(dataset_name)

    folder_path = os.path.dirname(id2allquery_path)
    os.makedirs(folder_path, exist_ok=True)

    with open(id2allquery_path, 'r') as f:
        id2allquery = json.load(f)
    with open(query2url_path, 'r') as f:
        query2url = json.load(f)
    with open(url2info_path, 'r') as f:
        url2info = json.load(f)

    id2infos = get_id2infos(test_data_path)
    sample_num = 0
    res_list = []
    __iter = tqdm(id2allquery.keys())
    for qid in __iter:
        queries = id2allquery[qid]

        for query in queries:
            if not query in query2url.keys():
                continue
            urls = query2url[query]
            for url in urls:
                if not url in url2info.keys():
                    continue
                infos = url2info[url]
                if "fetched_text_chunked_list" in infos.keys() and len(infos["fetched_text_chunked_list"]) > 0:
                    for cxt_idx, cxt in enumerate(infos["fetched_text_chunked_list"]):
                        question_id = "{}-@split@-{}-@split@-{}".format(qid, url, cxt_idx)
                        res_list.append(
                            {
                                "question_id": question_id,
                                "image": id2infos[qid]["image"],
                                "text": prompt.format(cxt, id2infos[qid]["question"]).replace('\\n', '\n'),
                            }
                        )
                        __iter.set_description("{}, totoal sample num = {}".format(dataset_name, len(res_list)))

    save_path = './intermediate_files/{}/segment_level_items.jsonl'.format(dataset_name)
    with open(save_path, 'w') as f:
        for tmp in res_list:
            json.dump(tmp, f)
            f.write('\n')

if __name__ == '__main__':
    main()