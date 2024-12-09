#!/bin/bash

gpu=0

# dataset_name in [UDK-VQA, UDK-VQA-20240905]
dataset_name=UDK-VQA

# for version in [raw, gt_segment]
# version=raw
# test_data_path='./datasets/test/'$dataset_name'/test_'$version'.jsonl'

# for version in [searchlvlms]
version=searchlvlms
test_data_path='./intermediate_files/'$dataset_name'/test_'$version'.jsonl'
##################################
# ↑↑      Need to modify      ↑↑ #
##################################




##################################
# ↓↓ No Modification Required ↓↓ #
##################################
test_img_dir='./datasets/test/'$dataset_name'/images'
prediction_path='./predictions/'$dataset_name'/{}_'$version'.json'

test_lvlm_list_llava15=llava_v1.5_7b
test_lvlm_list_llavanext=llava_next_mistral_7b
test_lvlm_list_monkey=monkey
test_lvlm_list_cogvlm_chat=cogvlm-chat
test_lvlm_list_qwen_chat=qwen_chat
test_lvlm_list_llava_v15_7b_xtuner=llava-v1.5-7b-xtuner
test_lvlm_list_MiniCPM_V2=MiniCPM-V2
test_lvlm_list_XComposer2=XComposer2
test_lvlm_list_MMAlaya=MMAlaya
test_lvlm_list_VisualGLM_6b=VisualGLM_6b
test_lvlm_list_mPLUG_Owl2=mPLUG-Owl2
test_lvlm_list_internvl=InternVL-Chat-V1-5


CUDA_VISIBLE_DEVICES=$gpu /cpfs01/user/lichuanhao/miniconda3/envs/llava_next/bin/python scripts/eval/eval_lvlms.py \
    --test_lvlm_list $test_lvlm_list_llavanext \
    --test_data_path $test_data_path \
    --test_img_dir $test_img_dir \
    --prediction_path $prediction_path
    
CUDA_VISIBLE_DEVICES=$gpu /cpfs01/user/lichuanhao/miniconda3/envs/py39/bin/python scripts/eval/eval_lvlms.py \
    --test_lvlm_list $test_lvlm_list_internvl \
    --test_data_path $test_data_path \
    --test_img_dir $test_img_dir \
    --prediction_path $prediction_path

CUDA_VISIBLE_DEVICES=$gpu /cpfs01/user/lichuanhao/miniconda3/envs/py39/bin/python scripts/eval/eval_lvlms.py \
    --test_lvlm_list $test_lvlm_list_monkey \
    --test_data_path $test_data_path \
    --test_img_dir $test_img_dir \
    --prediction_path $prediction_path
CUDA_VISIBLE_DEVICES=$gpu /cpfs01/user/lichuanhao/miniconda3/envs/py39/bin/python scripts/eval/eval_lvlms.py \
    --test_lvlm_list $test_lvlm_list_cogvlm_chat \
    --test_data_path $test_data_path \
    --test_img_dir $test_img_dir \
    --prediction_path $prediction_path
CUDA_VISIBLE_DEVICES=$gpu /cpfs01/user/lichuanhao/miniconda3/envs/py39/bin/python scripts/eval/eval_lvlms.py \
    --test_lvlm_list $test_lvlm_list_qwen_chat \
    --test_data_path $test_data_path \
    --test_img_dir $test_img_dir \
    --prediction_path $prediction_path
CUDA_VISIBLE_DEVICES=$gpu /cpfs01/user/lichuanhao/miniconda3/envs/py39/bin/python scripts/eval/eval_lvlms.py \
    --test_lvlm_list $test_lvlm_list_llava_v15_7b_xtuner \
    --test_data_path $test_data_path \
    --test_img_dir $test_img_dir \
    --prediction_path $prediction_path
CUDA_VISIBLE_DEVICES=$gpu /cpfs01/user/lichuanhao/miniconda3/envs/py39/bin/python scripts/eval/eval_lvlms.py \
    --test_lvlm_list $test_lvlm_list_MiniCPM_V2 \
    --test_data_path $test_data_path \
    --test_img_dir $test_img_dir \
    --prediction_path $prediction_path
CUDA_VISIBLE_DEVICES=$gpu /cpfs01/user/lichuanhao/miniconda3/envs/py39/bin/python scripts/eval/eval_lvlms.py \
    --test_lvlm_list $test_lvlm_list_XComposer2 \
    --test_data_path $test_data_path \
    --test_img_dir $test_img_dir \
    --prediction_path $prediction_path

CUDA_VISIBLE_DEVICES=$gpu /cpfs01/user/lichuanhao/miniconda3/envs/py39_trans433/bin/python scripts/eval/eval_lvlms.py \
    --test_lvlm_list $test_lvlm_list_MMAlaya \
    --test_data_path $test_data_path \
    --test_img_dir $test_img_dir \
    --prediction_path $prediction_path
CUDA_VISIBLE_DEVICES=$gpu /cpfs01/user/lichuanhao/miniconda3/envs/py39_trans433/bin/python scripts/eval/eval_lvlms.py \
    --test_lvlm_list $test_lvlm_list_VisualGLM_6b \
    --test_data_path $test_data_path \
    --test_img_dir $test_img_dir \
    --prediction_path $prediction_path

CUDA_VISIBLE_DEVICES=$gpu /cpfs01/user/lichuanhao/miniconda3/envs/mplug_owl2/bin/python scripts/eval/eval_lvlms.py \
    --test_lvlm_list $test_lvlm_list_mPLUG_Owl2 \
    --test_data_path $test_data_path \
    --test_img_dir $test_img_dir \
    --prediction_path $prediction_path