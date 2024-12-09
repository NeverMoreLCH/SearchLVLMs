import os
import sys
sys.path.append(os.environ.get("vlmevalkit_dir"))
from vlmeval.config import supported_VLM

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


import argparse
import json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test_lvlm_list', type=str, default='llava-v1.5-7b-xtuner')
    parser.add_argument('--test_data_path', type=str, default='./datasets/test/UDK-VQA/test_raw.jsonl')
    parser.add_argument('--test_img_dir', type=str, default='./datasets/test/UDK-VQA/images')
    parser.add_argument('--prediction_path', type=str, default='./predictions/{}/searchlvlms_{}.json')

    args = parser.parse_args()
    return args


def check_gt_ans(pred, answer_choice, answer_text):
    answers = [answer_choice, answer_choice[0], answer_text, answer_choice+' '+answer_text, answer_choice[0]+' '+answer_text]
    format_answers1 = ['The answer is {}'.format(x) for x in answers]
    format_answers2 = ['Answer: {}'.format(x) for x in answers]
    format_answers3 = ['{}</s>'.format(x) for x in answers]

    answers.extend(format_answers1)
    answers.extend(format_answers2)
    answers.extend(format_answers3)
    answers = list(set([x.lower().strip().strip('.') for x in answers]))
    pred = pred.lower().strip().strip('.')

    if pred in answers:
        return 1

    answers.remove(answer_choice[0].lower().strip().strip('.'))
    for gt_ans in answers:
        min_len = min(len(gt_ans), len(pred))
        trunc_pred = pred[: min_len]
        if gt_ans == trunc_pred:
            return 1

    format_answers1 = list(set([x.lower().strip().strip('.') for x in format_answers1]))
    for gt_ans in format_answers1:
        if gt_ans in pred:
            return 1
    return 0


def main():
    args = parse_args()

    test_data_path = args.test_data_path
    img_dir = args.test_img_dir

    test_lvlm_list = args.test_lvlm_list.split('@')
    candidate_list = [json.loads(q) for q in open(os.path.expanduser(test_data_path), "r")]
    dataset_name = os.path.dirname(test_data_path).split('/')[-1]

    print(len(candidate_list))
    print('test_data_path = {}'.format(test_data_path))
    print('prediction_path = {}'.format(args.prediction_path))

    for lvlm in test_lvlm_list:
        model = supported_VLM[lvlm]()
        lvlm_predictions = {}

        p_path = args.prediction_path.format(lvlm)
        folder_path = os.path.dirname(p_path)
        os.makedirs(folder_path, exist_ok=True)

        correct_num = 0
        __iter = tqdm(candidate_list)
        for idx, sample in enumerate(__iter):
            __iter.set_description(lvlm)
            question_id = sample["question_id"]
            answer_choice = sample["category"]
            answer_text = sample["answer_text"]
            image_name = sample["image"]
            img_path = os.path.join(img_dir, image_name)

            pred = model.generate(img_path, sample['text'])
            score = check_gt_ans(pred, answer_choice, answer_text)
            lvlm_predictions[question_id] = {"pred": pred, "score": score}
            correct_num = correct_num + lvlm_predictions[question_id]["score"]
        
        print("{}: acc = {}".format(lvlm, correct_num * 100 / len(candidate_list)))
        json.dump(lvlm_predictions, open(p_path, 'w'))

if __name__ == '__main__':
    main()