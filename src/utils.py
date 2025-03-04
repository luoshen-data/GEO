import json
import math
import itertools
from glob import glob
import time
from openai import OpenAI

client = OpenAI()

PROMPT_TEMPLATE = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_msg}[/INST]"

def get_prompt(source, query):
    system_prompt = """You are a helpful, respectful and honest assistant.
Given a web source, and context, your only purpose is to summarize the source, and extract topics that may be relevant to the context. Even if a line is distinctly relevant to the context, include that in the summary. It is preferable to pick chunks of text, instead of isolated lines.
"""

    user_msg = f"### Context: ```\n{query}\n```\n\n ### Source: ```\n{source}\n```\n Now summarize the text in more than 1000 words, keeping in mind the context and the purpose of the summary. Be as detailed as possible.\n"

    return PROMPT_TEMPLATE.format(system_prompt=system_prompt, user_msg=user_msg)

import re
import nltk



def get_num_words(line):
    return len([x for x in line if len(x)>2])

def extract_citations_new(text):
    def ecn(sentence):
        try:
            citation_pattern = r'\[[^\w\s]*\d+[^\w\s]*\]'
            citations = re.findall(citation_pattern, sentence)
            
            if not citations:
                return []
                
            result = []
            for citation in citations:
                try:
                    digits = re.findall(r'\d+', citation)
                    if digits:
                        result.append(int(digits[0]))
                except (ValueError, IndexError) as e:
                    print(f"Error parsing citation '{citation}': {str(e)}")
            return result
        except Exception as e:
            print(f"Error in citation extraction: {str(e)}")
            return []

    try:
        if text is None:
            print("Warning: Received None text in extract_citations_new")
            return []
            
        paras = re.split(r'\n\n', text)

        # Split each paragraph into sentences
        sentences = []
        for p in paras:
            try:
                sentences.append(nltk.sent_tokenize(p))
            except Exception as e:
                print(f"Error tokenizing paragraph: {str(e)}")
                sentences.append([p])  # Use the whole paragraph as a single sentence

        # Split each sentence into words
        words = []
        for sentence in sentences:
            sentence_words = []
            for s in sentence:
                try:
                    tokens = nltk.word_tokenize(s)
                    citations = ecn(s)
                    sentence_words.append((tokens, s, citations))
                except Exception as e:
                    print(f"Error processing sentence: {str(e)}")
                    # Add an empty entry to maintain structure
                    sentence_words.append(([], s, []))
            words.append(sentence_words)
            
        return words
    except Exception as e:
        print(f"Error in extract_citations_new: {str(e)}")
        return []

def impression_wordpos_count_simple(sentences, n = 5, normalize=True):
    try:
        if not sentences:
            print("Warning: Empty sentences in impression_wordpos_count_simple")
            return [1/n for _ in range(n)] if normalize else [0 for _ in range(n)]
            
        sentences = list(itertools.chain(*sentences))
        scores = [0 for _ in range(n)]
        
        for i, sent in enumerate(sentences):
            try:
                if not isinstance(sent, tuple) or len(sent) < 3:
                    continue
                    
                words, _, citations = sent
                
                if not citations:
                    continue
                    
                for cit in citations:
                    try:
                        if not isinstance(cit, int) or cit <= 0 or cit > n:
                            print(f"Citation out of range: {cit}")
                            continue
                            
                        score = get_num_words(words)
                        score *= math.exp(-1 * i / (len(sentences)-1)) if len(sentences)>1 else 1
                        score /= len(citations) if len(citations) > 0 else 1

                        scores[cit-1] += score
                    except Exception as e:
                        print(f"Error processing citation {cit}: {str(e)}")
            except Exception as e:
                print(f"Error processing sentence at position {i}: {str(e)}")
                
        if normalize:
            total = sum(scores)
            if total > 0:
                return [x/total for x in scores]
            else:
                return [1/n for _ in range(n)]
        else:
            return scores
    except Exception as e:
        print(f"Error in impression_wordpos_count_simple: {str(e)}")
        return [1/n for _ in range(n)] if normalize else [0 for _ in range(n)]

def impression_word_count_simple(sentences, n = 5, normalize=True):
    sentences = list(itertools.chain(*sentences))
    scores = [0 for _ in range(n)]
    for i, sent in enumerate(sentences):
        for cit in sent[2]:
            score = get_num_words(sent[0])
            score /= len(sent[2])
            try: scores[cit-1] += score
            except: print(f'Citation Hallucinated: {cit}')
    return [x/sum(scores) for x in scores] if normalize and sum(scores)!=0 else [1/n for _ in range(n)] if normalize else scores
    

def impression_pos_count_simple(sentences, n = 5, normalize=True):
    sentences = list(itertools.chain(*sentences))
    scores = [0 for _ in range(n)]
    for i, sent in enumerate(sentences):
        for cit in sent[2]:
            score = 1
            score *= math.exp(-1 * i / (len(sentences)-1)) if len(sentences)>1 else 1
            score /= len(sent[2])
            try: scores[cit-1] += score
            except: print(f'Citation Hallucinated: {cit}')
    return [x/sum(scores) for x in scores] if normalize and sum(scores)!=0 else [1/n for _ in range(n)] if normalize else scores
                

def impression_para_based(sentences, n = 5, normalize = True, alpha = 1.1, beta = 1.5, gamma2 = 1/math.e):
    scores = [0 for _ in range(n)]
    power_scores = [1 for _ in range(n)]
    average_lines = sum([len(x) for x in sentences])/len(sentences)
    for i, para in enumerate(sentences):
        citation_counts = [0 for _ in range(n)]
        for sent in para:
            for c in sent[2]:
                try:
                    citation_counts[c-1] += get_num_words(sent[0])
                except Exception as e:
                    print(f"Citation Hallucinated: {c}")
        if sum(citation_counts)==0:
            continue
        
        for cit_num, cit in enumerate(citation_counts):
            if cit==0: continue
            score = cit/sum(citation_counts)
            
            score *= beta**(len(para)/average_lines - 1)
            
            if i == 0:
                score *= 1
            elif i != len(sentences)-1:
                score *= math.exp(-1 * i / (len(sentences)-2))
            else:
                score *= gamma2

            try:
                power_scores[cit_num] *= (alpha) ** (cit/sum(citation_counts))
                scores[cit_num] += score
            except:
                print(f'Citation Hallucinated: {cit}')
         
    final_scores = [x*y for x, y in zip(scores, power_scores)]
    return [x/sum(final_scores) for x in final_scores] if normalize and sum(final_scores)!=0 else final_scores


subj_cache_file = None
def impression_subjpos_detailed(sentences, query, n = 5, normalize = True, idx = 0):
    return impression_subjective_impression(sentences, query, n = n, normalize = normalize, idx = idx, metric = 'subjpos_detailed')

def impression_diversity_detailed(sentences, query, n = 5, normalize = True, idx = 0):
    return impression_subjective_impression(sentences, query, n = n, normalize = normalize, idx = idx, metric = 'diversity_detailed')

def impression_uniqueness_detailed(sentences, query, n = 5, normalize = True, idx = 0):
    return impression_subjective_impression(sentences, query, n = n, normalize = normalize, idx = idx, metric = 'uniqueness_detailed')

def impression_follow_detailed(sentences, query, n = 5, normalize = True, idx = 0):
    return impression_subjective_impression(sentences, query, n = n, normalize = normalize, idx = idx, metric = 'follow_detailed')

def impression_influence_detailed(sentences, query, n = 5, normalize = True, idx = 0):
    return impression_subjective_impression(sentences, query, n = n, normalize = normalize, idx = idx, metric = 'influence_detailed')

def impression_relevance_detailed(sentences, query, n = 5, normalize = True, idx = 0):
    return impression_subjective_impression(sentences, query, n = n, normalize = normalize, idx = idx, metric = 'relevance_detailed')

def impression_subjcount_detailed(sentences, query, n = 5, normalize = True, idx = 0):
    return impression_subjective_impression(sentences, query, n = n, normalize = normalize, idx = idx, metric = 'subjcount_detailed')
    
def impression_subjective_impression(sentences, query, n = 5, normalize = True, idx = 0, metric = 'subjective_impression'):
    # print(hash((sentences, query, n, idx)))
    # 3/0
    def returnable_score_from_scores(scores):
        avg_score = sum(scores.values())/len(scores.values())
        if metric != 'subjective_impression':
            avg_score = scores[metric]
        return [avg_score if _==idx else 0 for _ in range(n)]

    
    global subj_cache_file
    cache_file = 'gpt-eval-scores-cache_new-new.json'

    if os.environ.get('SUBJ_STATIC_CACHE', None) is not None:
        if subj_cache_file is None:
            subj_cache_file = json.load(open(cache_file))
    else:
        if os.path.exists(cache_file):
            subj_cache_file = json.load(open(cache_file))
        else:
            subj_cache_file = dict()
            json.dump(subj_cache_file, open(cache_file, 'w'), indent=2)
    cache = subj_cache_file
    # TODO: Fix str(idx) issue
    # from pdb import set_trace
    if str((sentences, query)) in cache:
        if str(idx) in cache[str((sentences, query))]:
            print('Okay we have a hit!')
            # new_scores = []
            # for idx in range(5):
            #     sc = cache[str((sentences, query))][str(idx)]
            #     new_scores.append(sum(sc.values())/len(sc.values()))
            # return [x/sum(new_scores) for x in new_scores] if normalize else new_scores
            return returnable_score_from_scores(cache[str((sentences, query))][str(idx)])
    # TODO: If we don't have a hit, fine, just return 0 or something
    # set_trace()
    return [0 if _==idx else 0 for _ in range(n)]
    def convert_to_number(x, min_val = 1.0):
        try: return max(min(5, float(x)), min_val)
        except: return min_val
    scores = dict()
    for prompt_file in glob('geval_prompts/*.txt'):
        prompt = open(prompt_file).read()
        prompt = prompt.replace('[1]',f'[{idx+1}]')
        cur_prompt = prompt.format(query = query, answer = sentences)
        while True:
            try:
                _response = client.completions.create(
                    model='chatgpt-4o-latest',
                    prompt = cur_prompt,
                    temperature=0.0,
                    max_tokens=3,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    logprobs=5,
                    n=1
                )
                # print(_response.usage)
                # time.sleep(0.5)
                logprobs = _response['choices'][0]['logprobs']['top_logprobs'][0]
                total_sum = sum([((math.e)**v) for v in logprobs.values()])
                avg_score = sum([convert_to_number(k) * ((math.e)**v)/total_sum for k,v in logprobs.items()])
                scores[os.path.split(prompt_file)[-1].split('.')[0]] = avg_score
                break
            except Exception as e:
                print('Error in GPT-Eval', e)
                time.sleep(10)
    avg_score = sum(scores.values())/len(scores.values())
    cache = json.load(open(cache_file))
    if str((sentences, query)) not in cache:
        cache[str((sentences, query))] = dict()
    cache[str((sentences, query))][idx] = scores
    json.dump(cache, open(cache_file, 'w'), indent=2)
    return returnable_score_from_scores(scores)

import os
CACHE_FILE = os.environ.get('GLOBAL_CACHE_FILE', 'global_cache.json')
# CACHE_FILE = 'global_cache.json'

from search_try import search_handler
from generative_le import generate_answer

def check_summaries_exist(sources, summaries):
    for source in sources:
        s2 = [x['summary'] for x in source['sources']]  
        if s2 == summaries:
            return source
    return None

def get_answer(query, summaries = None, n = 5, num_completions = 1, cache_idx = 0, regenerate_answer = False, write_to_cache = True, loaded_cache = None):
    # print(CACHE_FILE, query)
    if loaded_cache is None:    cache = json.load(open(CACHE_FILE))
    else: cache = loaded_cache
    if summaries is None:
        if cache.get(query) is None:
            search_results = search_handler(query, source_count = n)
            if loaded_cache is None:    cache = json.load(open(CACHE_FILE))
            else: cache = loaded_cache
            cache[query] = [{'sources': search_results['sources'], 'responses': []}]
            if write_to_cache:
                json.dump(cache, open(CACHE_FILE, 'w'), indent=2)
        else:
            search_results = cache[query][cache_idx]

        summaries = [x['summary'] for x in search_results['sources']]
    cached_source = check_summaries_exist(cache[query], summaries)
    if not regenerate_answer and cached_source is not None:
        if len(cached_source['responses']) > 0:
            print('Cache Hit')
            answers = cached_source['responses'][-1]
            return cached_source
        else:
            answers = generate_answer(query, summaries, num_completions = num_completions) 
    else:
        answers = generate_answer(query, summaries, num_completions = num_completions) 
    ret_value = None
    # Update the cache
    if loaded_cache is None:    cache = json.load(open(CACHE_FILE))
    else: cache = loaded_cache

    if cache.get(query) is None:
        if summaries is None:
            cache[query] = [{'sources': search_results['sources'], 'responses': [answers]}]
        else:
            # Fix: Instead of creating minimal sources, use the standard format
            # Original: cache[query] = [{'sources': [{'summary' : x} for x in summaries], 'responses': [answers]}]
            # Generate proper source entries with all fields
            proper_sources = []
            for i, summary in enumerate(summaries):
                proper_sources.append({
                    'summary': summary,
                    'source': f"Source {i+1}",  # Default value if no real source is available
                    'url': f"Source URL {i+1}",  # Default value if no URL is available
                    'text': f"Summary: {summary}"  # Default format used in search_handler
                })
            cache[query] = [{'sources': proper_sources, 'responses': [answers]}]
    else:
        flag = False
        for source in cache[query]:
            s2 = [x['summary'] for x in source['sources']]  
            if s2 == summaries:
                source['responses'].append(answers)
                ret_value = source
                flag = True
                break
        if not flag:
            if summaries is None:
                cache[query].append({'sources': search_results['sources'], 'responses': [answers]})
            else:
                # Fix: Ensure full source information is maintained
                # Original: cache[query].append({'sources': [{'summary' : x, 'source' : y} for x, y in zip(summaries, cache[query][0]['sources'])], 'responses': [answers]})
                
                # First check if cache[query][0]['sources'] has complete source information
                existing_sources = cache[query][0]['sources']
                proper_sources = []
                
                for i, (summary, existing) in enumerate(zip(summaries, existing_sources)):
                    source_entry = {
                        'summary': summary,
                        # Use existing source information if available, otherwise use defaults
                        'source': existing.get('source', f"Source {i+1}"),
                        'url': existing.get('url', f"Source URL {i+1}"),
                        'text': existing.get('text', f"Summary: {summary}")
                    }
                    proper_sources.append(source_entry)
                
                cache[query].append({'sources': proper_sources, 'responses': [answers]})
    if write_to_cache:
        json.dump(cache, open(CACHE_FILE, 'w'), indent=2)

    return ret_value if ret_value is not None else cache[query][-1] 