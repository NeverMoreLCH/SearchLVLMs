import json
import os
from tqdm import tqdm
import random
import clip
import torch
import numpy as np
from sklearn.cluster import KMeans

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test_data_path', type=str, default='./datasets/test/UDK-VQA/test_raw.jsonl')
    parser.add_argument('--clip_dir', type=str, default='/cpfs01/user/lichuanhao/huggingface_cache/clip')
    
    parser.add_argument('--top_num', type=int, default=10)
    parser.add_argument('--device', type=str, default='gpu')

    parser.add_argument('--prompt', type=str, default="Given context: {}.\n\nQuestion: {}\nAnswers:\nA. {}\nB. {}\nC. {}\nD. {}\nE. No correct answers\n\nAnswer with the option's letter from the given choices directly based on the context and the image.")
    parser.add_argument('--prompt_nocxt', type=str, default="Question: {}\nAnswers:\nA. {}\nB. {}\nC. {}\nD. {}\nE. No correct answers\n\nAnswer with the option's letter from the given choices directly based on the context and the image.")

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


def get_id2predictions(read_path):

    id2predictions = {}
    tmp_pred = [json.loads(q) for q in open(os.path.expanduser(read_path), "r")]

    __iter = tqdm(tmp_pred)
    for infos in __iter:
        question_id = infos["question_id"]
        real_qid = question_id.split("-@split@-")[0]
        pred = infos["text"]
        cxt = infos["prompt"].split("\n\nContext: ")[-1].split("\nQuestion: ")[0]

        if real_qid in id2predictions.keys():
            id2predictions[real_qid].append({
                "question_id": question_id,
                "context": cxt,
                "pred": pred
            })
        else:
            id2predictions[real_qid]= [{
                "question_id": question_id,
                "context": cxt,
                "pred": pred
            }]

    return id2predictions


def _cluster(model, processor, device, texts, cluster_num):
    cluster_num = min(cluster_num, len(texts))
    if cluster_num == 0:
        return []

    ret_list = []
    text_features = []
    
    model = model.to(device)
    model = model.float()
    
    for text in texts:
        inputs = clip.tokenize(text, truncate=True).to(device) 
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


def main():

    args = parse_args()

    test_data_path = args.test_data_path
    dataset_name = os.path.dirname(test_data_path).split('/')[-1]
    segment_score_path = './intermediate_files/{}/segment_score.json'.format(dataset_name)

    id2infos = get_id2infos(test_data_path)
    id2predictions = get_id2predictions(segment_score_path)

    device = torch.device(args.device)
    model, processor = clip.load("ViT-B/32", download_root=args.clip_dir, device=device)

    res = []
    top_num = args.top_num
    __iter = tqdm(id2infos.keys())
    for idx, qid in enumerate(__iter):

        question = id2infos[qid]["question"]
        choices = id2infos[qid]["choices"]
        if qid in id2predictions.keys():
            context_list = id2predictions[qid]
            select_num = min(len(context_list), top_num*2)
            context_list = list([x["context"] for x in context_list])[:select_num]
            select_list = random.sample(context_list, select_num)
            res_context = _cluster(model, processor, device, select_list, top_num)
            text = args.prompt.format(res_context, question, choices[0], choices[1], choices[2], choices[3]).replace('\\n', '\n')
        else:
            text = args.prompt_nocxt.format(question, choices[0], choices[1], choices[2], choices[3]).replace('\\n', '\n')
        
        res.append({
            "question_id": qid,
            "image": id2infos[qid]["image"],
            "text": text,
            "category": "{}".format(id2infos[qid]["gt_choice"]),
            "answer_text": id2infos[qid]["gt_ans"]
        })

    save_path = './intermediate_files/{}/test_searchlvlms.jsonl'.format(dataset_name)
    with open(save_path, 'w') as f:
        for tmp in res:
            json.dump(tmp, f)
            f.write('\n')

if __name__ == '__main__':
    main()