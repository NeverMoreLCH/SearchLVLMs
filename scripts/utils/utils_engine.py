import os
from googleapiclient.discovery import build
import requests
from utils import get_hash_name_suffix, parse_ner_res
import urllib.parse
import json
import string


def longest_common_substring_3(s1, s2, s3):
    len1, len2, len3 = len(s1), len(s2), len(s3)
    dp = [[[0] * (len3+1) for _ in range(len2+1)] for __ in range(len1+1)]
    
    max_len = 0
    end_index_s1 = -1
    
    for i in range(1, len1+1):
        for j in range(1, len2+1):
            for k in range(1, len3+1):
                if s1[i-1] == s2[j-1] == s3[k-1]:
                    dp[i][j][k] = dp[i-1][j-1][k-1] + 1
                    if dp[i][j][k] > max_len:
                        max_len = dp[i][j][k]
                        end_index_s1 = i - 1
                else:
                    dp[i][j][k] = 0
    
    if max_len > 0:
        return s1[end_index_s1 - max_len + 1:end_index_s1 + 1].strip()
    else:
        return ""

def longest_common_substring_2(s1, s2):
    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    max_len = 0 
    end_index_s1 = -1
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_index_s1 = i - 1
            else:
                dp[i][j] = 0
    
    if max_len > 0:
        return s1[end_index_s1 - max_len + 1:end_index_s1 + 1].strip()
    else:
        return ""


def bing_visual_search(img_path, SUBSCRIPTION_KEY, site='', nlp=None):
    BASE_URI = 'https://api.bing.microsoft.com/v7.0/images/visualsearch'

    HEADERS = {'Ocp-Apim-Subscription-Key': SUBSCRIPTION_KEY, "Accept-Language": 'en'}
    file = {'image' : ('myfile', open(img_path, 'rb'))}
    params = {"mkt": "en-US"}

    try:
        try:
            response = requests.post(BASE_URI, headers=HEADERS, params=params, files=file)
        except Exception as err:
            return str(err)
        response.raise_for_status()
        response_json = response.json()

        brq_set = set()
        entity_set = set()
        related_list = []
        url_title_set = set()
        for tag in response_json['tags']:
            for act in tag["actions"]:
                if "actionType" in act.keys() and act["actionType"] == "BestRepresentativeQuery" and "displayName" in act.keys():
                    brq_set.add(act["displayName"])
                if "actionType" in act.keys() and act["actionType"] == "Entity" and "displayName" in act.keys():
                    entity_set.add(act["displayName"])
                if "actionType" in act.keys() and act["actionType"] == "RelatedSearches":
                    if "data" in act.keys() and "value" in act["data"].keys():
                        data = act["data"]["value"]
                        for result in data:
                            if not result['text'] in related_list:
                                related_list.append(result['text'])
                if "actionType" in act.keys() and act["actionType"] in ["PagesIncluding", "VisualSearch"]:
                    if "data" in act.keys() and "value" in act["data"].keys():
                        data = act["data"]["value"]
                        for result in data:
                            title = result["name"] if "name" in result.keys() else ""
                            url_title_set.add(title)

        brq_list = list(brq_set)
        entity_list = list(entity_set)
        url_title_list = list(url_title_set)

        search_str = ''
        if len(brq_list) > 0:
            search_str = ', '.join(brq_list)
        elif len(entity_list) > 0:
            search_str = ', '.join(entity_list)
        elif len(related_list) >= 3:
            str1, str2, str3 = related_list[0], related_list[1], related_list[2]
            search_str = longest_common_substring_3(str1, str2, str3)
        
        if search_str == '' and len(related_list) == 2:
            str1, str2 = related_list[0], related_list[1]
            search_str = longest_common_substring_2(str1, str2)
        if search_str == '' and len(related_list) == 1:
            search_str = related_list[0]
        
        if search_str == '' and len(url_title_list) >= 3:
            str1, str2, str3 = url_title_list[0], url_title_list[1], url_title_list[2]
            search_str = longest_common_substring_3(str1, str2, str3)
        if search_str == '' and len(url_title_list) == 2:
            str1, str2 = url_title_list[0], url_title_list[1]
            search_str = longest_common_substring_2(str1, str2)
        if search_str == '' and len(url_title_list) == 1:
            title = url_title_list[0]
            parse_ent_list = [x['entity'] for x in parse_ner_res(title, nlp(title))]
            search_str = ', '.join(parse_ent_list)

        ret_dict = {'search_str': search_str, 'brq_list':brq_list, 'entity_list':entity_list, 'related_list':related_list, 'url_title_list':url_title_list}
        return ret_dict

    except Exception as err:
        return str(err)

def bing_text_search(search_term, subscription_key, site='', freshness=''):

    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    query = search_term
    if len(site) > 0:
        query = query + ' ' + site

    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    if freshness == '':
        params = {"q": query, "count": 10, "mkt": "en-US"}
    else:
        params = {"q": query, "count": 10, "mkt": "en-US", 'freshness': freshness}

    try:
        try:
            response = requests.get(endpoint, headers=headers, params=params)
        except Exception as err:
            return str(err)
        response.raise_for_status()
        search_results = response.json()
        
        ret = []
        for result in search_results.get("webPages", {}).get("value", []):
            title = result["name"] if "name" in result.keys() else ""
            snippet = result["snippet"] if "snippet" in result.keys() else ""
            url = result["url"] if "url" in result.keys() else ""
            datePublished = result["datePublished"] if "datePublished" in result.keys() else "0000-00-00T00:00:00.0000000Z"
            primaryImageOfPage = result["primaryImageOfPage"] if "primaryImageOfPage" in result.keys() else ""
            item_type = "bing_text_search"
            
            ret.append({"title": title, "url": url, "snippet": snippet, "datePublished": datePublished, "primaryImageOfPage": primaryImageOfPage, "type": item_type})
        return ret
    except Exception as err:
        return str(err)

def bing_news_search(search_term, subscription_key, site='', exclude='', freshness=''):

    endpoint = 'https://api.bing.microsoft.com/v7.0/news/search'
    query = search_term
    if len(site) > 0:
        query = query + ' ' + site
    if len(exclude) > 0:
        query = query + ' ' + exclude

    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    if freshness == '':
        params = {"q": query, "count": 10, "mkt": "en-US"}
    else:
        params = {"q": query, "count": 10, "mkt": "en-US", 'freshness': freshness}

    try:
        try:
            response = requests.get(endpoint, headers=headers, params=params)
        except Exception as err:
            return str(err)

        response.raise_for_status()
        search_results = response.json()

        ret = []
        for result in search_results.get("value", []):
            title = result["name"] if "name" in result.keys() else ""
            snippet = result["description"] if "description" in result.keys() else ""
            url = result["url"] if "url" in result.keys() else ""
            datePublished = result["datePublished"] if "datePublished" in result.keys() else "0000-00-00T00:00:00.0000000Z"
            primaryImageOfPage = result["primaryImageOfPage"] if "primaryImageOfPage" in result.keys() else ""
            item_type = "bing_news_search"
            
            ret.append({"title": title, "url": url, "snippet": snippet, "datePublished": datePublished, "primaryImageOfPage": primaryImageOfPage, "type": item_type})
        return ret
    except Exception as err:
        return str(err)

def bing_image_search(search_term, subscription_key, site=''):
    endpoint = "https://api.bing.microsoft.com/v7.0/images/search"

    query = search_term
    if len(site) > 0:
        query = query + ' ' + site

    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": query, "count": 10,  "imageType": "photo"}

    try:
        try:
            response = requests.get(endpoint, headers=headers, params=params)
        except Exception as err:
            return str(err)
        response.raise_for_status()
        search_results = response.json()

        ret = []
        for item in search_results.get("value", []):
            web_url = item['hostPageUrl'] if 'hostPageUrl' in item.keys() else ''
            web_title = item['title'] if 'title' in item.keys() else ''
            web_snippet = item['snippet'] if 'snippet' in item.keys() else ''

            try:
                img_url = item['contentUrl']
            except:
                try:
                    img_url = item['thumbnailUrl']
                except:
                    img_url = 'Error'
            img_name, parse_suffix = get_hash_name_suffix(img_url)
            try:
                img_suffix = item['encodingFormat']
            except:
                img_suffix = parse_suffix
            img_name = 'bing_' + img_name

            tmp_dict = {'img_url': img_url, 'img_name': img_name, 'img_suffix': img_suffix,
                        'web_url': web_url, 'web_title': web_title, 'web_snippet': web_snippet}
            ret.append(tmp_dict)
        return ret

    except Exception as err:
        return str(err)

def google_search(search_term, api_key, cse_id, site=''):
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        results = service.cse().list(q=search_term, cx=cse_id).execute()
        if isinstance(results, str):
            return results
        return results['items']
    except Exception as err:
        return str(err)

def google_text_search(search_term, api_key, cse_id, site=''):
    try:
        query = search_term
        if len(site) > 0:
            query = query + ' ' + site

        try:
            results = google_search(query, api_key, cse_id, site)
        except Exception as err:
            return str(err)
        if isinstance(results, str):
            return results
        if isinstance(results, dict) and 'error' in results.keys():
            return results['error']['message']

        ret = []
        for result in results:
            title = result["title"] if "title" in result.keys() else ""
            snippet = result["snippet"] if "snippet" in result.keys() else ""
            url = result["link"] if "link" in result.keys() else ""
            datePublished = result["datePublished"] if "datePublished" in result.keys() else "0000-00-00T00:00:00.0000000Z"
            primaryImageOfPage = result["primaryImageOfPage"] if "primaryImageOfPage" in result.keys() else ""
            item_type = "google_text_search"

            ret.append({"title": title, "url": url, "snippet": snippet, "datePublished": datePublished, "primaryImageOfPage": primaryImageOfPage, "type": item_type})
        return ret
    except Exception as err:
        return str(results) + str(err)

def google_image_search(search_term, api_key, cse_id, site=''):
    try:
        query = search_term
        if len(site) > 0:
            query = query + ' ' + site
        encoded_query = urllib.parse.quote(query)
        search_url = 'https://www.googleapis.com/customsearch/v1?q={}&cx={}&searchType=image&key={}'.format(encoded_query, cse_id, api_key)
        results_raw = requests.get(search_url)
        results = results_raw.json()
    except Exception as err:
        return str(results) + str(err)
    if isinstance(results, str):
        return results
    if isinstance(results, dict) and 'error' in results.keys():
        return results['error']['message']

    ret = []
    for item in results["items"]:
        if "image" in item.keys() and "contextLink" in item["image"].keys():
            web_url = item["image"]['contextLink']
        else:
            web_url = ' '
        web_title = item['title'] if 'snititleppet' in item.keys() else ''
        web_snippet = item['snippet'] if 'snippet' in item.keys() else ''

        try:
            img_url = item['link']
        except:
            try:
                img_url = item['image']['thumbnailLink']
            except:
                img_url = 'Error'
        img_name, parse_suffix = get_hash_name_suffix(img_url)
        img_suffix = item['fileFormat'].split('image/')[-1] if 'fileFormat' in item.keys() else ''
        img_suffix = parse_suffix if img_suffix == '' else img_suffix

        img_name = 'google_' + img_name
        tmp_dict = {'img_url': img_url, 'img_name': img_name, 'img_suffix': img_suffix,
                    'web_url': web_url, 'web_title': web_title, 'web_snippet': web_snippet}
        ret.append(tmp_dict)
    return ret

def search_from_web(search_term, engine='google', search_type='text', site_list=[], exclude_list=[], freshness='', nlp=None):
    google_api_key = os.environ.get("google_api_key")
    google_text_cse_id = os.environ.get("google_text_cse_id")
    google_image_cse_id = os.environ.get("google_image_cse_id")

    bing_text_api_key = os.environ.get("bing_text_api_key")
    bing_img_api_key = os.environ.get("bing_img_api_key")
    bing_visual_api_key = os.environ.get("bing_visual_api_key")

    site_list = ['site:{}'.format(x) for x in site_list]
    site_str = ' OR '.join(site_list)

    exclude_list = ['-site:{}'.format(x) for x in exclude_list]
    exclude_str = ' '.join(exclude_list)

    if search_type == 'text':
        if engine == 'google':
            return google_text_search(search_term, google_api_key, google_text_cse_id, site_str)
        elif engine == 'bing':
            return bing_text_search(search_term, bing_text_api_key, site_str, freshness=freshness)
    
    elif search_type == 'image':
        if engine == 'google':
            return google_image_search(search_term, google_api_key, google_image_cse_id, site_str)
        elif engine == 'bing':
            return bing_image_search(search_term, bing_img_api_key, site_str)

    elif search_type == 'news':
        return bing_news_search(search_term, bing_text_api_key, site_str, exclude_str, freshness=freshness)
    
    elif search_type == 'visual': 
        return bing_visual_search(search_term, bing_visual_api_key, site_str, nlp=nlp)

