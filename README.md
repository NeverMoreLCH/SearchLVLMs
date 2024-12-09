# SearchLVLMs
**SearchLVLMs: A Plug-and-Play Framework for Augmenting Large Vision-Language Models by Searching Up-to-Date Internet Knowledge (NeurIPS 2024)**  
Chuanhao Li, Zhen Li, Chenchen Jing, Shuo Liu, Wenqi Shao, Yuwei Wu, Ping Luo, Yu Qiao, Kaipeng Zhang  
[[Homepage]](https://nevermorelch.github.io/SearchLVLMs.github.io/) [[Paper]](https://arxiv.org/pdf/2405.14554)

![Example Image](https://github.com/NeverMoreLCH/SearchLVLMs.github.io/blob/main/static/images/framework.png?raw=true)

<br>

## News
- 2024.12.09: 🎉 The inference code and UDK-VQA dataset are released!
- 2024.09.26: 🎉 SearchLVLMs is accepted by NeurIPS 2024!

<br>

## Install
```
conda env create -f environment.yml
conda activate searchlvlms
```

<br>

## Prerequisites
#### Llama3
Install [Llama3](https://github.com/meta-llama/llama3/) and download the [checkpoint](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).

#### VLMEvalKit
Install [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) and download the checkpoints of LVLMs for testing.

#### LLaVA-1.5
Install [LLaVA-1.5](https://github.com/haotian-liu/LLaVA) and download the [pretrained model](https://huggingface.co/liuhaotian/llava-v1.5-7b/tree/main) and the [projector weights](https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/tree/main).

#### NER
Download the NER model via [huggingface](https://huggingface.co/dslim/bert-large-NER).

#### CLIP
Download the CLIP model via [huggingface](https://huggingface.co/docs/transformers/model_doc/clip).

<br>

## UDK-VQA Dataset and Checkpoint
For both UDK-VQA and the checkpoint of our filter, download them from:
[[Google Drive](https://drive.google.com/drive/folders/1qt_ttxcY43AvC17Xcv8ZnsH0fMNxa9LE?usp=sharing)] or
[[Baidu NetDisk (password: DSPS)](https://pan.baidu.com/s/1XCJq9mSItZAd21fY0Xz1zQ)]

Unzip the zip files and make sure the file structure looks like this:
```
SearchLVLMs
----checkpoints
--------llava_lora_content_filter
--------llava_lora_website_filter
----datasets
--------test
--------train
...
```

<br>

## Configurations
Configure the variables in `scripts/init_env_variable.sh`, `scripts/iag.sh` and `scripts/eval.sh`.
- `scripts/init_env_variable.sh`
```
# used for generating queries for the question via llama3.
llama3_dir="<your path of llama3 project>"

# used for running eval.sh
vlmevalkit_dir="<your path of vlmevalkit project>"

# used for calling gpt to generate queries for the question.
OPENAI_API_KEY=""
OPENAI_ENDPOINT=""

# the keys of the google search engine are optional, as we mainly use the bing search engine.
google_api_key=""
google_text_cse_id=""
google_image_cse_id=""

# img_api is optional, as it's used for generating samples, which is not released yet.
bing_text_api_key=""
bing_img_api_key=""
bing_visual_api_key=""
```
For the variables in `scripts/iag.sh` and `scripts/eval.sh`, you can easily understand them via their names.

<br>

## Evaluation
You can run the following scripts to evaluate LVLMs (or LVLMs+SearchLVLMs)
```
cd SearchLVLMs

# Active environment variable
source scripts/init_env_variable.sh

# Run SearchLVLMs to find the best context for each sample in the test set.
sh scripts/iag.sh

# Eval the accuracy of LVLMs (or LVLMs+SearchLVLMs) on the test set.
sh scripts/eval.sh

# Deactivate environment variable
source scripts/unset_env_variable.sh
```

<br>

## Citation
If any part of our paper and code is helpful to your work, please generously cite with:
```
@inproceedings{li2024searchlvlms,
  title={SearchLVLMs: A Plug-and-Play Framework for Augmenting Large Vision-Language Models by Searching Up-to-Date Internet Knowledge},
  author={Li, Chuanhao and Li, Zhen and Jing, Chenchen and Liu, Shuo and Shao, Wenqi and Wu, Yuwei and Luo, Ping and Qiao, Yu and Zhang, Kaipeng},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```
