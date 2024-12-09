#!/bin/bash

device=cuda:0
dataset_name=UDK-VQA

project_dir="<your path of SearchLVLMs>"
llava_dir="<your path of LLaVA-1.5 project>"

ner_model_path="<your path of models--dslim--bert-large-NER>"
llama3_ckpt_dir="<your path of Meta-Llama-3-8B-Instruct>"
llama3_tokenizer_path="<your path of Meta-Llama-3-8B-Instruct/tokenizer.model>"
clip_dir="<your path of clip>"
##################################
# ↑↑      Need to modify      ↑↑ #
##################################




##################################
# ↓↓ No Modification Required ↓↓ #
##################################
test_data_path='./datasets/test/'$dataset_name'/test_raw.jsonl'
test_img_dir='./datasets/test/'$dataset_name'/images'

save_step_for_step1=10
save_step_for_step2=500
save_step_for_step3=500
top_num=10

prompt_for_segment_selection="How helpful is this context in answering the question based on the image? Choose the best option.\n\nContext: {}\nQuestion: {}\nOptions:\nA. 1.0\nB. 0.8\nC. 0.6\nD. 0.4\nE. 0.2\nF. 0.0\n"
prompt_for_question_answering="Given context: {}.\n\nQuestion: {}\nAnswers:\nA. {}\nB. {}\nC. {}\nD. {}\nE. No correct answers\n\nAnswer with the option's letter from the given choices directly based on the context and the image."
prompt_for_question_answering_nocxt="Question: {}\nAnswers:\nA. {}\nB. {}\nC. {}\nD. {}\nE. No correct answers\n\nAnswer with the option's letter from the given choices directly based on the context and the image."

llava_ckp_name=llava_lora_content_filter
llava_ckp_dir=$project_dir'/checkpoints/'$llava_ckp_name
question_filepath=$project_dir'/intermediate_files/'$dataset_name'/segment_level_items.jsonl'
answer_filepath=$project_dir'/intermediate_files/'$dataset_name'/segment_score.json'

/cpfs01/user/lichuanhao/miniconda3/envs/test_env/bin/python scripts/iag/step1_gen_search_queries.py \
    --test_data_path $test_data_path \
    --test_img_dir $test_img_dir \
    --ner_model_path $ner_model_path \
    --llama3_ckpt_dir $llama3_ckpt_dir \
    --llama3_tokenizer_path $llama3_tokenizer_path \
    --clip_dir $clip_dir \
    --save_step $save_step_for_step1 \
    --device $device

/cpfs01/user/lichuanhao/miniconda3/envs/test_env/bin/python scripts/iag/step2_call_search_engines.py \
    --test_data_path $test_data_path \
    --save_step $save_step_for_step2

/cpfs01/user/lichuanhao/miniconda3/envs/test_env/bin/python scripts/iag/step3_crawl_from_urls.py \
    --test_data_path $test_data_path \
    --save_step $save_step_for_step3

/cpfs01/user/lichuanhao/miniconda3/envs/test_env/bin/python scripts/iag/step4_gen_data4hierarchical_filter.py \
    --test_data_path $test_data_path \
    --prompt "$prompt_for_segment_selection"

# step5
/cpfs01/user/lichuanhao/miniconda3/envs/llava/bin/python $llava_dir'llava/eval/model_vqa.py' \
    --model-base lmsys/vicuna-7b-v1.5 \
    --model-path $llava_ckp_dir \
    --question-file $question_filepath \
    --image-folder $test_img_dir \
    --answers-file $answer_filepath

/cpfs01/user/lichuanhao/miniconda3/envs/test_env/bin/python scripts/iag/step6_gen_test_searchlvlms.py \
    --test_data_path $test_data_path \
    --clip_dir $clip_dir \
    --top_num $top_num \
    --device $device \
    --prompt "$prompt_for_question_answering" \
    --prompt_nocxt "$prompt_for_question_answering_nocxt"
