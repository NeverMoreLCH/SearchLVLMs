import os
import sys
import hashlib
import requests
from openai import OpenAI
from newspaper import Article

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def hash_url(url: str) -> str:
    sha256_hash = hashlib.sha256()
    sha256_hash.update(url.encode('utf-8'))
    hex_dig = sha256_hash.hexdigest()
    return hex_dig

def get_hash_name_suffix(url):
    name = hash_url(url)
    suffix = name.split('.')[-1]
    if 'jpg' in suffix:
        suffix = 'jpg'
    elif 'jpeg' in suffix:
        suffix = 'jpeg'
    elif 'png' in suffix:
        suffix = 'png'
    elif 'gif' in suffix:
        suffix = 'gif'
    elif 'webp' in suffix:
        suffix = 'webp'
    elif 'svg' in suffix:
        suffix = 'svg'
    else:
        suffix = 'jpg'
    return name, suffix

def text_chunk(text, token_num_per_chunk=800):
    text = text.replace('\n\n', ' ').replace('\n', ' ')
    text_list = text.split('.')

    ret = []
    tmp_len = 0
    tmp_text_list = []
    for para in text_list:
        if len(para) == 0:
            continue
        now_len = len(para.split(' '))
        if now_len > token_num_per_chunk:
            continue

        if tmp_len + now_len <= token_num_per_chunk:
            tmp_str = para.strip() + '.'
            tmp_text_list.append(tmp_str)
            tmp_len = tmp_len + now_len
        else:
            ret.append(' '.join(tmp_text_list))
            tmp_len = 0
            tmp_text_list = []

    if tmp_len > 0:
        ret.append(' '.join(tmp_text_list))

    return ret
        
def check_fetched_text(text):
    good_flag = True
    if len(text) < 200:
        good_flag = False
    if len(text.split(' ')) <= 40:
        good_flag = False
    if 'Error' in text[:60]:
        good_flag = False
    if 'cookies' in text[:100]:
        good_flag = False
    if 'Cookies' in text[:100]:
        good_flag = False
    if text.startswith('Please upgrade your browser'):
        good_flag = False
    if 'Please make sure your browser supports JavaScript and cookies' in text:
        good_flag = False
    return good_flag

def parse_ner_res(question, ner_res):
    ret_list = []
    save_start_idx = 0
    save_end_idx = 0
    last_type = ""
    for info in ner_res:
        ent_type = info["entity"]
        start_idx = info["start"]
        end_idx = info["end"]

        if "-" in ent_type:
            prefix, suffix = ent_type.split("-")
        else:
            prefix = "O"
            suffix = "O"

        if prefix in ["B", "O"]:
            if not last_type == "":
                ret_list.append({"entity":question[save_start_idx:save_end_idx], "type":last_type})
            save_start_idx = start_idx
            save_end_idx = end_idx
        if prefix == "I":
            save_end_idx = end_idx

        last_type = suffix

    if not last_type == "":
        ret_list.append({"entity":question[save_start_idx:save_end_idx], "type":last_type})

    return ret_list


def gpt_text_only(text, pair_num=1, temperature=0.1, model='gpt-3.5-turbo', OPENAI_API_KEY='', OPENAI_ENDPOINT=''):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_ENDPOINT")
    )

    if pair_num > 1:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": text,
                }
            ],
            model=model,
            temperature = temperature,
            n = pair_num
        )
    else:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": text,
                }
            ],
            model=model,
            temperature = temperature
        )
    res = [str(c.message.content) for c in response.choices]
    return res


def fetch_by_newspaper(url):
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-us;q=0.5,en;q=0.3',
        'Cache-Control': 'max-age=0',
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0'
    }
    try:
        response = requests.get(url, headers=headers, timeout=(3, 6)) # headers=headers
        html_content = response.text
        article = Article(url='')
        article.set_html(html_content)
        article.parse()
    except Exception as err:
        return {'fetched_title': 'Error', 'fetched_text': 'Error Fetch! ' + str(err)}
    try:
        fetched_title = article.title
    except Exception as err:
        fetched_title = 'Error Title! ' + str(err)

    try:
        fetched_text = article.text
    except Exception as err:
        fetched_text = 'Error Fetch! ' + str(err)

    ret_dict = {'fetched_title': fetched_title, 'fetched_text': fetched_text}
    return ret_dict

