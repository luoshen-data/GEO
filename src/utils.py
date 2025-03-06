import json
import math
import itertools
from glob import glob
import time
import os
from openai import OpenAI
import re
import nltk
import traceback

# Enable this to get more detailed debugging information
DEBUG_MODE = os.environ.get('DEBUG_MODE', 'False').lower() in ('true', '1', 't')

# Initialize global variables
client = OpenAI()
CACHE_FILE = os.environ.get('GLOBAL_CACHE_FILE', 'global_cache.json')

def debug_log(message):
    """Log debug messages when DEBUG_MODE is enabled"""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")

# Initialize the cache before any function tries to access it
def initialize_cache():
    # Initialize main cache
    cache_file = os.environ.get('GLOBAL_CACHE_FILE', 'global_cache.json')
    if not os.path.exists(cache_file):
        debug_log(f"Cache file {cache_file} not found. Creating a new empty cache file.")
        print(f"Cache file {cache_file} not found. Creating a new empty cache file.")
        try:
            with open(cache_file, 'w') as f:
                json.dump({}, f)
        except Exception as e:
            print(f"Error creating cache file: {str(e)}")
            return False
    
    # Initialize subjective impression cache
    subj_cache_file_path = 'gpt-eval-scores-cache_new-new.json'
    if not os.path.exists(subj_cache_file_path):
        debug_log(f"Subjective cache file {subj_cache_file_path} not found. Creating a new empty cache file.")
        print(f"Subjective cache file {subj_cache_file_path} not found. Creating a new empty cache file.")
        try:
            with open(subj_cache_file_path, 'w') as f:
                json.dump({}, f)
        except Exception as e:
            print(f"Error creating subjective cache file: {str(e)}")
            return False
            
    return True

# Initialize cache at module import time
initialize_cache()

PROMPT_TEMPLATE = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_msg}[/INST]"

def get_prompt(source, query):
    system_prompt = """You are a helpful, respectful and honest assistant.
Given a web source, and context, your only purpose is to summarize the source, and extract topics that may be relevant to the context. Even if a line is distinctly relevant to the context, include that in the summary. It is preferable to pick chunks of text, instead of isolated lines.
"""

    user_msg = f"### Context: ```\n{query}\n```\n\n ### Source: ```\n{source}\n```\n Now summarize the text in more than 1000 words, keeping in mind the context and the purpose of the summary. Be as detailed as possible.\n"

    return PROMPT_TEMPLATE.format(system_prompt=system_prompt, user_msg=user_msg)

def get_num_words(line):
    return len([x for x in line if len(x)>2])

def extract_citations_new(text):
    def ecn(sentence):
        try:
            if not isinstance(sentence, str):
                print(f"Warning: Non-string input to citation extractor: {type(sentence)}")
                if sentence is None:
                    return []
                sentence = str(sentence)
                
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
        # Handle None or non-string input
        if text is None:
            print("Warning: Received None text in extract_citations_new")
            return []
            
        if not isinstance(text, str):
            print(f"Warning: Non-string input to extract_citations_new: {type(text)}")
            try:
                text = str(text)
            except Exception as e:
                print(f"Error converting input to string: {str(e)}")
                return []
                
        # Handle empty text
        if not text.strip():
            print("Warning: Received empty text in extract_citations_new")
            return []
            
        # Check if NLTK data is available and download if necessary
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("NLTK punkt tokenizer not found. Attempting to download...")
            try:
                nltk.download('punkt', quiet=True)
            except Exception as nltk_err:
                print(f"Error downloading NLTK data: {str(nltk_err)}")
                # Fall back to simple sentence splitting
                sentences = [[[([], text, [])]]]
                return sentences
                
        # Split text into paragraphs
        paras = re.split(r'\n\n', text)
        if not paras:
            paras = [text]  # Use the whole text as one paragraph if splitting fails

        # Split each paragraph into sentences
        sentences = []
        for p in paras:
            try:
                # Check if paragraph is a string
                if not isinstance(p, str):
                    p = str(p)
                if not p.strip():
                    continue  # Skip empty paragraphs
                
                # Use NLTK's sentence tokenizer
                sent_tokens = nltk.sent_tokenize(p)
                if not sent_tokens:
                    sent_tokens = [p]  # Use the paragraph as a single sentence if tokenizing fails
                sentences.append(sent_tokens)
            except Exception as e:
                print(f"Error tokenizing paragraph: {str(e)}")
                # Use the whole paragraph as a single sentence
                if p and p.strip():
                    sentences.append([p])

        # If we couldn't extract any sentences, return an empty result
        if not sentences:
            print("Warning: No sentences could be extracted from the text")
            return []

        # Process each sentence to extract words and citations
        words = []
        for sentence_list in sentences:
            sentence_words = []
            for s in sentence_list:
                try:
                    if not s or not isinstance(s, str):
                        continue
                        
                    # Use NLTK's word tokenizer
                    tokens = nltk.word_tokenize(s)
                    if not tokens:
                        tokens = s.split()  # Fall back to simple splitting
                        
                    # Extract citations
                    citations = ecn(s)
                    sentence_words.append((tokens, s, citations))
                except Exception as e:
                    print(f"Error processing sentence: {str(e)}")
                    # Add a simple entry to maintain structure
                    if s and isinstance(s, str):
                        sentence_words.append((s.split(), s, []))
            
            if sentence_words:  # Only add non-empty lists
                words.append(sentence_words)
            
        # If we couldn't process any words, return a default structure
        if not words:
            print("Warning: No words could be processed from the text")
            return [[([], text, [])]]
            
        return words
    except Exception as e:
        print(f"Error in extract_citations_new: {str(e)}")
        # Return a minimal valid structure
        return [[([], text if isinstance(text, str) else "Error processing text", [])]]

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

    # Ensure cache exists
    initialize_cache()
    
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
    if not sources or not summaries:
        return None
        
    try:
        for source in sources:
            try:
                # Use get() to safely access 'sources' key and provide a default empty list
                source_list = source.get('sources', [])
                
                # Use a safer approach to extract summaries
                s2 = []
                for x in source_list:
                    try:
                        s2.append(x.get('summary', ''))
                    except Exception:
                        # If we can't get the summary for some reason, use an empty string
                        s2.append('')
                
                # Compare the summaries
                if s2 == summaries:
                    return source
            except Exception as e:
                # If processing a specific source fails, just skip it and continue
                print(f"Error processing a source in check_summaries_exist: {str(e)}")
                continue
    except Exception as e:
        print(f"Error in check_summaries_exist: {str(e)}")
        
    return None

def get_answer(query, summaries = None, n = 5, num_completions = 1, cache_idx = 0, regenerate_answer = False, write_to_cache = True, loaded_cache = None):
    debug_log(f"get_answer called with query: '{query}', n={n}, num_completions={num_completions}")
    
    try:
        # Ensure cache exists
        try:
            initialize_cache()
        except Exception as cache_init_err:
            print(f"Error initializing cache: {str(cache_init_err)}")
            debug_log(f"Cache initialization error: {traceback.format_exc()}")
            # Continue execution despite cache initialization error
        
        # Check if cache file exists and create it if it doesn't
        try:
            if not os.path.exists(CACHE_FILE):
                print(f"Cache file {CACHE_FILE} not found. Creating a new empty cache file.")
                with open(CACHE_FILE, 'w') as f:
                    json.dump({}, f)
                    
            if loaded_cache is None:
                try:
                    cache = json.load(open(CACHE_FILE))
                    debug_log(f"Loaded cache from {CACHE_FILE}, contains {len(cache)} entries")
                except json.JSONDecodeError as json_err:
                    print(f"Error decoding JSON from cache file: {str(json_err)}. Creating a new cache.")
                    debug_log(f"JSON decode error: {traceback.format_exc()}")
                    cache = {}
                except Exception as cache_load_err:
                    print(f"Error loading cache file: {str(cache_load_err)}. Creating a new cache.")
                    debug_log(f"Cache load error: {traceback.format_exc()}")
                    cache = {}
            else: 
                cache = loaded_cache
                debug_log("Using provided loaded_cache")
        except Exception as cache_err:
            print(f"Error handling cache file: {str(cache_err)}. Using an empty cache.")
            debug_log(f"Cache handling error: {traceback.format_exc()}")
            cache = {}
            
        if summaries is None:
            debug_log("No summaries provided, will fetch from search or cache")
            try:
                if cache.get(query) is None:
                    debug_log(f"Query '{query}' not found in cache, performing search")
                    try:
                        search_results = search_handler(query, source_count = n)
                        debug_log(f"Search completed, found {len(search_results.get('sources', []))} sources")
                    except Exception as search_err:
                        print(f"Error in search_handler: {str(search_err)}")
                        debug_log(f"Search error: {traceback.format_exc()}")
                        # Create default search results
                        search_results = {
                            'sources': [{'summary': f"Failed to retrieve source {i+1}", 
                                        'source': f"Source {i+1}", 
                                        'url': f"URL {i+1}", 
                                        'text': f"Failed to retrieve text {i+1}"} for i in range(n)]
                        }
                    
                    try:
                        if loaded_cache is None:
                            cache = json.load(open(CACHE_FILE))
                        else:
                            cache = loaded_cache
                    except Exception as reload_err:
                        print(f"Error reloading cache after search: {str(reload_err)}")
                        debug_log(f"Cache reload error: {traceback.format_exc()}")
                        # Continue with current cache
                    
                    cache[query] = [{'sources': search_results['sources'], 'responses': []}]
                    
                    if write_to_cache:
                        try:
                            json.dump(cache, open(CACHE_FILE, 'w'), indent=2)
                            debug_log(f"Wrote updated cache to {CACHE_FILE}")
                        except Exception as write_err:
                            print(f"Error writing to cache file: {str(write_err)}")
                            debug_log(f"Cache write error: {traceback.format_exc()}")
                else:
                    debug_log(f"Query '{query}' found in cache, using cached results")
                    search_results = cache[query][cache_idx]

                try:
                    summaries = [x.get('summary', f"Missing summary for source {i+1}") for i, x in enumerate(search_results['sources'])]
                    debug_log(f"Extracted {len(summaries)} summaries from search results")
                except Exception as summary_err:
                    print(f"Error extracting summaries from search results: {str(summary_err)}")
                    debug_log(f"Summary extraction error: {traceback.format_exc()}")
                    summaries = [f"Default summary {i+1}" for i in range(n)]
            except Exception as summaries_err:
                print(f"Error handling summaries: {str(summaries_err)}")
                debug_log(f"Summaries handling error: {traceback.format_exc()}")
                summaries = [f"Default summary {i+1}" for i in range(n)]
        else:
            debug_log(f"Using {len(summaries)} provided summaries")
        
        try:
            cached_source = check_summaries_exist(cache.get(query, []), summaries)
            if cached_source:
                debug_log("Found matching summaries in cache")
            else:
                debug_log("No matching summaries found in cache")
        except Exception as cache_check_err:
            print(f"Error in check_summaries_exist: {str(cache_check_err)}")
            debug_log(f"Cache check error: {traceback.format_exc()}")
            cached_source = None
            
        try:
            if not regenerate_answer and cached_source is not None:
                if len(cached_source.get('responses', [])) > 0:
                    print('Cache Hit')
                    debug_log("Cache hit - returning cached response")
                    answers = cached_source['responses'][-1]
                    return cached_source
                else:
                    debug_log("Cache entry found but no responses, generating new answer")
                    answers = generate_answer(query, summaries, num_completions = num_completions) 
            else:
                debug_log("Generating new answer")
                try:
                    answers = generate_answer(query, summaries, num_completions = num_completions)
                    debug_log(f"Answer generation successful, received {len(answers)} responses")
                except Exception as gen_err:
                    print(f"Error in generate_answer: {str(gen_err)}")
                    debug_log(f"Answer generation error: {traceback.format_exc()}")
                    # Create default answers
                    answers = ["Failed to generate response due to an error."] * num_completions
        except Exception as answer_gen_err:
            print(f"Error handling answer generation: {str(answer_gen_err)}")
            debug_log(f"Answer handling error: {traceback.format_exc()}")
            answers = ["Failed to generate response due to an error."] * num_completions
            
        ret_value = None
        
        # Update the cache
        debug_log("Updating cache with new response")
        try:
            if loaded_cache is None:
                try:
                    cache = json.load(open(CACHE_FILE))
                except Exception as reload_err:
                    print(f"Error reloading cache for update: {str(reload_err)}")
                    debug_log(f"Cache reload error: {traceback.format_exc()}")
                    # Continue with current cache
            else:
                cache = loaded_cache

            if cache.get(query) is None:
                debug_log(f"Query '{query}' not in cache, creating new entry")
                if summaries is None:
                    cache[query] = [{'sources': search_results['sources'], 'responses': [answers]}]
                else:
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
                debug_log(f"Query '{query}' found in cache, updating entry")
                flag = False
                for source in cache[query]:
                    try:
                        s2 = [x.get('summary', '') for x in source.get('sources', [])]
                        if s2 == summaries:
                            debug_log("Found matching summaries entry, appending new response")
                            source['responses'].append(answers)
                            ret_value = source
                            flag = True
                            break
                    except Exception as source_err:
                        print(f"Error processing source in cache: {str(source_err)}")
                        debug_log(f"Source processing error: {traceback.format_exc()}")
                        continue
                        
                if not flag:
                    debug_log("No matching summaries found, adding new entry")
                    if summaries is None:
                        cache[query].append({'sources': search_results['sources'], 'responses': [answers]})
                    else:
                        # Ensure full source information is maintained
                        existing_sources = cache[query][0].get('sources', [])
                        proper_sources = []
                        
                        try:
                            for i, summary in enumerate(summaries):
                                source_entry = {
                                    'summary': summary,
                                    # Use default values to ensure we have all necessary fields
                                    'source': f"Source {i+1}",
                                    'url': f"Source URL {i+1}",
                                    'text': f"Summary: {summary}"
                                }
                                # Try to use existing source information if available
                                if i < len(existing_sources):
                                    source_entry['source'] = existing_sources[i].get('source', source_entry['source'])
                                    source_entry['url'] = existing_sources[i].get('url', source_entry['url'])
                                    source_entry['text'] = existing_sources[i].get('text', source_entry['text'])
                                
                                proper_sources.append(source_entry)
                        except Exception as source_creation_err:
                            print(f"Error creating proper sources: {str(source_creation_err)}")
                            debug_log(f"Source creation error: {traceback.format_exc()}")
                            # Use simplified source creation as fallback
                            proper_sources = [{'summary': s, 'source': f"Source {i+1}", 'url': f"URL {i+1}", 'text': f"Text {i+1}"} 
                                            for i, s in enumerate(summaries)]
                        
                        cache[query].append({'sources': proper_sources, 'responses': [answers]})
                        
            if write_to_cache:
                try:
                    json.dump(cache, open(CACHE_FILE, 'w'), indent=2)
                    debug_log(f"Successfully wrote updated cache to {CACHE_FILE}")
                except Exception as write_err:
                    print(f"Error writing updated cache to file: {str(write_err)}")
                    debug_log(f"Cache write error: {traceback.format_exc()}")
        except Exception as cache_update_err:
            print(f"Error updating cache: {str(cache_update_err)}")
            debug_log(f"Cache update error: {traceback.format_exc()}")

        debug_log("Returning response from get_answer")
        if ret_value is not None:
            return ret_value
        else:
            try:
                return cache[query][-1]
            except Exception as return_err:
                print(f"Error returning from cache: {str(return_err)}")
                debug_log(f"Return error: {traceback.format_exc()}")
                # Return a default response structure
                default_sources = []
                for i in range(n):
                    default_sources.append({
                        'summary': f"Failed to retrieve summary {i+1} for query: {query}",
                        'source': f"Source {i+1}",
                        'url': f"Source URL {i+1}",
                        'text': f"Failed to retrieve text for query: {query}"
                    })
                
                default_response = {
                    'sources': default_sources,
                    'responses': [["Failed to generate response due to an error."] * 5]
                }
                return default_response
                
    except Exception as e:
        print(f"Error in get_answer: {str(e)}")
        debug_log(f"Top-level get_answer error: {traceback.format_exc()}")
        # Create a stack trace for better debugging
        traceback.print_exc()
        
        # Return a default response structure instead of None
        default_sources = []
        for i in range(n):
            default_sources.append({
                'summary': f"Failed to retrieve summary {i+1} for query: {query}",
                'source': f"Source {i+1}",
                'url': f"Source URL {i+1}",
                'text': f"Failed to retrieve text for query: {query}"
            })
        
        # Ensure we have a properly formatted response that won't cause errors in improve
        default_response = {
            'sources': default_sources,
            'responses': [["Failed to generate response due to an error."] * 5]  # Ensure we have 5 responses as expected by improve
        }
        return default_response 