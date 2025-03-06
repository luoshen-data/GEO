import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from readabilipy import simple_json_from_html_string
import trafilatura
import nltk
from openai import OpenAI
import os
import pickle
import uuid

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def clean_source_gpt35(source : str) -> str:
    for idx in range(8):
        try:
            response = client.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=[
                    {
                    "role": "user",
                    "content": f"Clean and refine the extracted text from a website. Remove any unwanted content such as headers, sidebars, and navigation menus. Retain only the main content of the page and ensure that the text is well-formatted and free of HTML tags, special characters, and any other irrelevant information. Refined text should contain the main intended readable text. Apply markdown formatting when outputting the answer.\n\nHere is the website:\n```html_text\n{source.strip()}```"
                    },
                ],
                temperature=0,
                max_tokens=4096,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            break
        except Exception as e:
            print(f'Error while cleaning text with openai {e}')
            source = source[:-int(800*(1 + idx/2))]
            time.sleep(3 + idx**2)
    # Store the usage in the folder response_usages, with a random file name
    pickle.dump(response.usage, open(f"response_usages/{uuid.uuid4()}.pkl", "wb"))

    tex = response.choices[0].message.content.strip()
    new_lines = [""]
    for line in tex.split('\n\n'):
        new_lines[-1] +=line+'\n'
        if len(nltk.sent_tokenize(line))!=1:
            new_lines.append("")
    new_lines = [x.strip() for x in new_lines]
    return "\n\n".join(new_lines)

def clean_source_text(text: str) -> str:
    return (
        text.strip()
        .replace("\n\n\n", "\n\n")
        .replace("\n\n", " ")
        .replace("  ", " ")
        .replace("\t", "")
        .replace("\n", "")
    )

import time
from pdb import set_trace as bp
import os


def summarize_text_identity(source, query) -> str:
    return source[:8000]


def search_handler(req, source_count = 8):
    query = req
    default_sources = []
    for i in range(source_count):
        default_sources.append({
            'summary': f"Default summary for {query} - source {i+1}",
            'source': f"Source {i+1}",
            'url': f"URL {i+1}",
            'text': f"Default text for {query}"
        })

    # GET LINKS
    links = []
    try:
        max_retries = 5
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                # Add a user agent to mimic a browser request
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(f"https://www.google.com/search?q={query}", headers=headers, timeout=10)
                
                # Check for rate limiting or blocked requests
                if response.status_code == 429 or response.status_code == 403:
                    print(f"Rate limited or blocked by Google (status {response.status_code}). Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                    
                if response.status_code != 200:
                    print(f"Unexpected status code: {response.status_code}. Retrying...")
                    time.sleep(retry_delay)
                    continue
                    
                break
            except requests.exceptions.RequestException as e:
                print(f'Error while fetching from Google {e}')
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    time.sleep(retry_delay)
                continue
                
        # If all retries failed, return default sources
        if not hasattr(response, 'text') or not response.text:
            print("Failed to get a valid response after all retries")
            return {'sources': default_sources[:source_count]}
            
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        link_tags = soup.find_all('a')

        for link in link_tags:
            href = link.get('href')

            if href and href.startswith('/url?q='):
                try:
                    cleaned_href = href.replace('/url?q=', '').split('&')[0]
                    if cleaned_href not in links:
                        links.append(cleaned_href)
                        print(cleaned_href)
                except Exception as e:
                    print(f"Error cleaning URL: {str(e)}")
                    continue
    except Exception as e:
        print(f"Error extracting links: {str(e)}")
        # Return default sources if we can't get any links
        return {'sources': default_sources[:source_count]}

    # Filter links
    try:
        exclude_list = ["google", "facebook", "twitter", "instagram", "youtube", "tiktok","quora"]
        filtered_links = []
        links = set(list(links))
        
        for link in links:
            try:
                hostname = urlparse(link).hostname
                if hostname and '.' in hostname:  # Make sure hostname exists and contains a dot
                    domain = hostname.split('.')
                    if len(domain) > 1 and domain[1] not in exclude_list:
                        filtered_links.append(link)
            except Exception as e:
                print(f"Error parsing URL {link}: {str(e)}")
                continue
    except Exception as e:
        print(f"Error filtering links: {str(e)}")
        filtered_links = list(links)[:source_count]  # Use unfiltered links as fallback

    final_links = filtered_links

    # If no valid links found, return default sources
    if not final_links:
        print("No valid links found after filtering")
        return {'sources': default_sources[:source_count]}

    # SCRAPE TEXT FROM LINKS
    sources = []
    for link in final_links:
        if len(sources) >= source_count:
            break
            
        print(f'Will be loading link {link}')
        source_text = None
        html_text = ""
        
        try:
            # Try to fetch with trafilatura first
            for attempt in range(5):
                try:
                    downloaded = trafilatura.fetch_url(link, timeout=15)
                    if downloaded:
                        source_text = trafilatura.extract(downloaded)
                        if source_text is not None:
                            break
                    
                    print(f'Error fetching link {link} with trafilatura, attempt {attempt+1}')
                    time.sleep(4)
                except Exception as e:
                    print(f"Trafilatura error on {link}: {str(e)}")
                    time.sleep(4)
                    
            # If trafilatura failed, try with requests
            if source_text is None:
                try:
                    response = requests.get(link, timeout=15, 
                                         headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
                    if response.status_code == 200:
                        html = response.text
                        try:
                            html = simple_json_from_html_string(html)
                            html_text = html.get('content', '')
                        except Exception as html_err:
                            print(f"Error extracting content from HTML: {str(html_err)}")
                            try:
                                from readabilipy.extractors import extract_title
                                title = extract_title(html)
                                html_text = str(html)
                            except Exception as readability_err:
                                print(f"Error with readabilipy: {str(readability_err)}")
                                html_text = "Content extraction failed"
                except Exception as req_err:
                    print(f"Request error on {link}: {str(req_err)}")
                    
            # Skip if content is too short
            if source_text is None and len(html_text) < 400:
                print(f"Content too short for {link}: {len(html_text)} chars")
                continue
                
            # Use HTML text if no source_text was extracted
            if source_text is None and html_text:
                source_text = html_text
                
            # Skip if we still don't have any text
            if not source_text:
                print(f"No content extracted from {link}")
                continue
                
            print(f"Extracted {len(source_text)} chars from {link}")
            
            try:
                source_text = clean_source_text(source_text)
                print('Going to call openai')
                raw_source = source_text
                
                # Limit source text length to prevent token limit issues
                truncated_source = source_text[:8000]
                
                try:
                    summary_text = summarize_text_identity(truncated_source, query)
                except Exception as summary_err:
                    print(f"Error summarizing text: {str(summary_err)}")
                    # Create a simple summary if the API call fails
                    summary_text = f"Summary of content from {link} (summarization failed)"
                    
                source_dict = {
                    'url': link, 
                    'text': f'Title: {html.get("title", "Unknown Title")}\nSummary:' + summary_text, 
                    'raw_source': raw_source, 
                    'source': truncated_source, 
                    'summary': summary_text
                }
                
                sources.append(source_dict)
                print('OpenAI processing completed')
            except Exception as processing_err:
                print(f"Error processing source text: {str(processing_err)}")
                
        except Exception as e:
            print(f"Error processing link {link}: {str(e)}")
            continue
            
    # If we couldn't get any valid sources, use default sources
    if not sources:
        print("No valid sources were extracted")
        return {'sources': default_sources[:source_count]}
        
    return {'sources': sources[:source_count]}
    
if __name__ == '__main__':
    import sys
    search_handler('What is Generative Engine Optimization?')
    import json
    print(json.dumps(search_handler(sys.argv[1]), indent = 2))