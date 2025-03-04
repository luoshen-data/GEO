from utils import get_answer, extract_citations_new, impression_subjective_impression, impression_wordpos_count_simple, impression_subjpos_detailed, impression_diversity_detailed, impression_uniqueness_detailed, impression_follow_detailed, impression_influence_detailed, impression_relevance_detailed, impression_subjcount_detailed, impression_pos_count_simple, impression_word_count_simple
from typing import List, Tuple
import numpy as np
import json
from geo_functions import *
import sys
import time
import os
from datasets import load_dataset

def identity(summary, source=None):
	return summary

IMPRESSION_FNS = {
	'simple_wordpos' : impression_wordpos_count_simple, 
	'simple_word' : impression_word_count_simple,
	'simple_pos' : impression_pos_count_simple,
	'subjective_score' : impression_subjective_impression,
	'subjpos_detailed' : impression_subjpos_detailed,
	'diversity_detailed' : impression_diversity_detailed,
	'uniqueness_detailed' : impression_uniqueness_detailed,
	'follow_detailed' : impression_follow_detailed,
	'influence_detailed' : impression_influence_detailed,
	'relevance_detailed' : impression_relevance_detailed,
	'subjcount_detailed' : impression_subjcount_detailed,
}


GEO_METHODS = {
	'identity' : identity,
	'fluent_gpt' : fluent_optimization_gpt,
	'unique_words_gpt' : unique_words_optimization_gpt,
	'authoritative_mine' : authoritative_optimization_mine,
	'more_quotes_mine' : more_quotes_mine,
	'citing_credible_mine': citing_credible_sources_mine,
	'simple_language_mine': simple_language_mine,
	'technical_terms_mine' : technical_terms_mine,
	'stats_optimization_gpt' : stats_optimization_mine,
	'seo_optimize_mine2' : seo_optimize_mine2,
}

EXTRACTIVE = False
loaded_cache = None
LAST_UPDATE_TIME = time.time()

def improve(query : str, idx : int, sources : List[str] = None, summaries : List[str] = None, impression_fn = impression_wordpos_count_simple, returnFullData = False, static_cache=os.environ.get('STATIC_CACHE', None)=='True') -> Tuple[np.array, List]: 
	global loaded_cache
	global LAST_UPDATE_TIME
	
	# Input validation
	if query is None or not isinstance(query, str) or query.strip() == "":
		print(f"Error: Invalid query: {query}")
		if returnFullData:
			return np.array([]), []
		else:
			return np.array([]), []
			
	if idx is None or not isinstance(idx, int) or idx < 0:
		print(f"Error: Invalid index: {idx}")
		if returnFullData:
			return np.array([]), []
		else:
			return np.array([]), []
	
	if static_cache:
		try:
			if loaded_cache is not None:
				modified_time = os.path.getmtime(os.environ.get('GLOBAL_CACHE_FILE', 'global_cache.json'))
				if modified_time - LAST_UPDATE_TIME > 0:
					loaded_cache = json.load(open(os.environ.get('GLOBAL_CACHE_FILE', 'global_cache.json')))
				LAST_UPDATE_TIME = modified_time
			else:
				loaded_cache = json.load(open(os.environ.get('GLOBAL_CACHE_FILE', 'global_cache.json')))
		except Exception as e:
			print(f"Error loading cache: {str(e)}")
			loaded_cache = None
	else:
		loaded_cache = None
		
	# idx indicates the website to boost
	print('query is', query)
	
	try:
		answers = get_answer(query, summaries=summaries, num_completions=5, n=5, loaded_cache=loaded_cache)
	except Exception as e:
		print(f"Error in get_answer: {str(e)}")
		if returnFullData:
			return np.array([]), []
		else:
			return np.array([]), []
	
	if sources is None:
		try:
			sources = [x['source'] for x in answers['sources']]
		except Exception as e:
			print(f"Error extracting sources: {str(e)}")
			if returnFullData:
				return np.array([]), []
			else:
				return np.array([]), []
				
	if summaries is None:
		try:
			summaries = [x['summary'] for x in answers['sources']]
		except Exception as e:
			print(f"Error extracting summaries: {str(e)}")
			if returnFullData:
				return np.array([]), []
			else:
				return np.array([]), []
	
	# Check if idx is valid
	if idx < 0 or idx >= len(summaries):
		print(f"Error: Index {idx} is out of range for summaries list with length {len(summaries)}")
		if returnFullData:
			return np.array([]), []
		else:
			return np.array([]), []

	try:
		answers = answers['responses'][-1]
	except Exception as e:
		print(f"Error extracting responses: {str(e)}")
		if returnFullData:
			return np.array([]), []
		else:
			return np.array([]), []

	try:
		if impression_fn == impression_subjective_impression or impression_fn == impression_subjpos_detailed or impression_fn == impression_diversity_detailed or impression_fn == impression_uniqueness_detailed or impression_fn == impression_follow_detailed or impression_fn == impression_influence_detailed or impression_fn == impression_relevance_detailed or impression_fn == impression_subjcount_detailed:
			orig_init_scores = np.array([impression_fn(x, query, 5, idx=idx) for x in answers])
			orig_init_scores = orig_init_scores[~np.all(orig_init_scores == 0, axis=1)]
		else:
			orig_init_scores = np.array([impression_fn(extract_citations_new(x), 5) for x in answers])
		
		init_scores = orig_init_scores.mean(axis=0)
		print('Init Scores: ', init_scores)
		improvements = []
		all_final_scores = []
	except Exception as e:
		print(f"Error calculating initial scores: {str(e)}")
		if returnFullData:
			return np.array([]), []
		else:
			return np.array([]), []

	for meth_name in GEO_METHODS:
		try:
			summaries_copy = summaries[:idx] + [GEO_METHODS[meth_name](summaries[idx])] + summaries[idx+1:] 
			answers = get_answer(query, summaries=summaries_copy, num_completions=5, n=5, loaded_cache=loaded_cache)
			answers = answers['responses'][-1]
			
			if impression_fn == impression_subjective_impression or impression_fn == impression_subjpos_detailed or impression_fn == impression_diversity_detailed or impression_fn == impression_uniqueness_detailed or impression_fn == impression_follow_detailed or impression_fn == impression_influence_detailed or impression_fn == impression_relevance_detailed or impression_fn == impression_subjcount_detailed:
				final_scores = np.array([impression_fn(x, query, 5, idx=idx) for x in answers])
				final_scores = final_scores[~np.all(final_scores == 0, axis=1)]
			else:
				final_scores = [impression_fn(extract_citations_new(x), 5) for x in answers]
				
			all_final_scores.append(np.array(final_scores))
			final_scores = np.array(final_scores).mean(axis=0)
			print(final_scores)
			improvements.append((final_scores - init_scores))
		except Exception as e:
			print(f"Error processing method {meth_name}: {str(e)}")
			continue
			
	if not improvements:
		print("No improvements were calculated successfully")
		if returnFullData:
			return np.array([]), []
		else:
			return np.array([]), []
			
	improvements = np.vstack(improvements)

	if returnFullData:
		return orig_init_scores, all_final_scores
	else:
		return improvements, improvements[:, idx] > 0


if __name__ == '__main__':
	dataset = load_dataset("GEO-Optim/geo-bench", 'test')
	# Run only 100 test cases in the dataset
	test_count = 0
	max_samples = 100  # Limit to 100 samples
	
	for i, k in enumerate(dataset['test']):
		try:
			print(f"\nProcessing item {i}: '{k['query']}'")
			
			# Check if sources and summaries exist
			if 'sources' not in k or not k['sources']:
				print(f"Error: No sources found for item {i}")
				continue
				
			# Extract sources and summaries from the dataset
			sources = [source['url'] for source in k['sources'] if 'url' in source]
			summaries = [source['cleaned_text'] for source in k['sources'] if 'cleaned_text' in source]
			
			# Check if sources and summaries were extracted successfully
			if not sources or not summaries:
				print(f"Error: Failed to extract sources or summaries for item {i}")
				continue
				
			# Check if sugg_idx exists and is valid
			if 'sugg_idx' not in k:
				print(f"Error: No sugg_idx found for item {i}")
				continue
				
			try:
				sugg_idx = int(k['sugg_idx'])
			except (ValueError, TypeError):
				print(f"Error: Invalid sugg_idx '{k['sugg_idx']}' for item {i}")
				continue
			
			if len(summaries) <= sugg_idx:
				print(f"Error: Index {sugg_idx} is out of range for summaries list with length {len(summaries)}")
				continue
				
			# Pass the sources and summaries to the improve function
			try:
				improvements, positive_improvements = improve(
					k['query'], 
					idx=sugg_idx, 
					sources=sources,
					summaries=summaries,
					impression_fn=impression_wordpos_count_simple
				)
				
				if len(improvements) > 0:
					print(improvements, positive_improvements)
				else:
					print("Skipping due to empty results")
					
				test_count += 1
				print(f"Completed test case {test_count}")
				
				# Break after processing max_samples
				if test_count >= max_samples:
					print(f"\nReached limit of {max_samples} samples. Stopping.")
					break
					
			except Exception as e:
				print(f"Error in improve function for item {i}: {str(e)}")
				import traceback
				traceback.print_exc()
				continue
				
		except Exception as e:
			print(f"Error processing item {i}: {str(e)}")
			import traceback
			traceback.print_exc()
			test_count += 1
			
			# Break after processing max_samples
			if test_count >= max_samples:
				print(f"\nReached limit of {max_samples} samples. Stopping.")
				break
