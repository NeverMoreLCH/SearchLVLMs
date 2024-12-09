# conda env
conda env create -f environment.yml
conda activate searchlvlms

# configurations
Llama3
https://github.com/meta-llama/llama3/
https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

VLMEvalKit
https://github.com/open-compass/VLMEvalKit

LLaVA
https://github.com/haotian-liu/LLaVA
https://huggingface.co/liuhaotian/llava-v1.5-7b/tree/main
https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/tree/main

NER
https://huggingface.co/dslim/bert-large-NER

CLIP
https://huggingface.co/docs/transformers/model_doc/clip


# download our datasets and checkpoints
链接：https://pan.baidu.com/s/1XCJq9mSItZAd21fY0Xz1zQ 
提取码：DSPS

download and unzip it to the datasets folder

datasets/train
samples for train website filter and content filter

datasets/test
two versions of UDK-VQA test set

# settings
init_env_variable.sh
iag.sh
eval.sh

# run
cd SearchLVLMs
source scripts/init_env_variable.sh
sh scripts/iag.sh
sh scripts/eval.sh
source scripts/unset_env_variable.sh