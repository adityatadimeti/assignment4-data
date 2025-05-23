import gzip
import os
import mmh3
from collections import defaultdict
import random
import re
from fastwarc.warc import ArchiveIterator, WarcRecordType
import resiliparse
from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
import unicodedata


import fasttext

def extract_text(input):
    """
    A function that takes a byte string containing HTML and returns a string containing the extracted text.
    """
    encoding_info = detect_encoding(input)
    try:
        html_string = input.decode(encoding_info, errors='ignore')
    except:
        html_string = input.decode('utf-8', errors='ignore')
    plain_text = extract_plain_text(html_string)
    return plain_text

def classify_text(text, model_path):
   text = text.replace('\n', ' ')
   model = fasttext.load_model(model_path)
   
   label, score = model.predict(text)
   label = label[0].replace("__label__", "")
   score = score[0]
   return label, score

def identify_language(input):
   return classify_text(input, "lid.176.bin")

def extract_random_examples(warc_path, num_examples=20):
    total_records = 0
    with open(warc_path, 'rb') as f:
        for record in ArchiveIterator(f):
            if record.record_type == WarcRecordType.response:
                total_records += 1
    
    indices_to_extract = random.sample(range(total_records), min(num_examples, total_records))
    extracted_records = []
    current_index = 0
    
    with open(warc_path, 'rb') as f:
        for record in ArchiveIterator(f):
            if record.record_type == WarcRecordType.response:
                if current_index in indices_to_extract:
                    content = record.reader.read()
                    text = extract_text(content)
                    extracted_records.append(text)
                        
                    if len(extracted_records) >= num_examples:
                        break
                
                current_index += 1
    
    return extracted_records

def mask_email(input):
    email_regex = r"""[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"""
    matches = re.findall(email_regex, input, re.IGNORECASE)
    count = len(matches)

    masked_string = re.sub(email_regex, "|||EMAIL_ADDRESS|||", input, flags=re.IGNORECASE)
    return masked_string, count

def mask_phone_numbers(text):
    phone_pattern = r'''(?:(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?:\s?(?:ext|x|ext\.)\s?\d{1,5})?)'''
    
    pattern = re.compile(phone_pattern, re.VERBOSE)
    
    matches = pattern.findall(text)
    count = len(matches)
    masked_text = pattern.sub("|||PHONE_NUMBER|||", text)
    return masked_text, count

def mask_ip_addresses(text):
   pattern = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
   matches = re.findall(pattern, text)
   masked_text = re.sub(pattern, "|||IP_ADDRESS|||", text)
   return masked_text, len(matches)

def test_classifier(warc_path, classifier_fn, name="classifier"):
   extracted_records = extract_random_examples(warc_path=warc_path)
   print(f"Testing {name} on {len(extracted_records)} records")
   results = [classifier_fn(record) for record in extracted_records]
   print(f"{name.capitalize()} results:", results)
   return extracted_records, results

def test_language_identification(warc_path):
   return test_classifier(warc_path, identify_language, "language identification")

def test_safety_classification(warc_path):
    # Extract records once and reuse for both classifiers
    extracted_records = extract_random_examples(warc_path=warc_path)
    print(f"Testing safety classifiers on {len(extracted_records)} records")
    
    # Run both classifiers on the same records
    nsfw_results = [nsfw(record) for record in extracted_records]
    toxic_results = [toxic(record) for record in extracted_records]
    
    print("NSFW results:", nsfw_results)
    print("Toxicity results:", toxic_results)
    
    # Combine results for analysis
    safety_results = []
    for i, record in enumerate(extracted_records):
        safety_results.append({
            "text_excerpt": record,
            "nsfw": nsfw_results[i],
            "toxic": toxic_results[i]
        })
    
    return safety_results

def nsfw(text):
   return classify_text(text, "/data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin")

def toxic(text):
   return classify_text(text, "/data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin")

def gopher(text):
    if not text:
        return False
    
    words = word_tokenize(text)
    lines = text.splitlines()
    
    word_count = len(words)
    if word_count < 50 or word_count > 100000:
        return False
    
    mean_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    if mean_word_length < 3 or mean_word_length > 10:
        return False
    
    if lines:
        ellipsis_percentage = sum(1 for line in lines if line.strip().endswith("...")) / len(lines)
        if ellipsis_percentage > 0.3:
            return False
    
    alpha_percentage = sum(1 for word in words if any(c.isalpha() for c in word)) / word_count if word_count > 0 else 0
    if alpha_percentage < 0.8:
        return False
    
    return True

def deduplicate_lines(input_paths, output_dir):
    line_counts = defaultdict(int)
    
    for path in input_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line_hash = mmh3.hash(line)
                line_counts[line_hash] += 1
    
    os.makedirs(output_dir, exist_ok=True)
    
    for path in input_paths:
        output_path = os.path.join(output_dir, os.path.basename(path))
        with open(path, 'r', encoding='utf-8') as infile:
            with open(output_path, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    line_hash = mmh3.hash(line)
                    if line_counts[line_hash] == 1:
                        outfile.write(line)
                        
def normalize_text(text):
    text = unicodedata.normalize('NFD', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return text.strip()

def get_ngrams(text, n):
    words = text.split()
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

def compute_minhash_signature(ngrams, num_hashes):
    signature = []
    for i in range(num_hashes):
        min_hash = float('inf')
        for ngram in ngrams:
            hash_val = mmh3.hash(ngram, seed=i) & 0x7FFFFFFF
            min_hash = min(min_hash, hash_val)
        signature.append(min_hash if ngrams else 0)
    return signature

def compute_jaccard_similarity(ngrams1, ngrams2):
    set1 = set(ngrams1)
    set2 = set(ngrams2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0

def minhash_deduplication(input_paths, num_hashes, num_bands, ngram_length, threshold, output_dir):
    rows_per_band = num_hashes // num_bands
    
    documents = []
    signatures = []
    
    for path in input_paths:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            normalized = normalize_text(content)
            ngrams = get_ngrams(normalized, ngram_length)
            signature = compute_minhash_signature(set(ngrams), num_hashes)
            documents.append((path, content, ngrams))
            signatures.append(signature)
    
    buckets = defaultdict(list)
    for doc_idx, signature in enumerate(signatures):
        for band_idx in range(num_bands):
            band_start = band_idx * rows_per_band
            band_end = band_start + rows_per_band
            band = tuple(signature[band_start:band_end])
            bucket_key = (band_idx, band)
            buckets[bucket_key].append(doc_idx)
    
    candidate_pairs = set()
    for bucket_docs in buckets.values():
        if len(bucket_docs) > 1:
            for i in range(len(bucket_docs)):
                for j in range(i + 1, len(bucket_docs)):
                    pair = tuple(sorted([bucket_docs[i], bucket_docs[j]]))
                    candidate_pairs.add(pair)
    
    duplicate_graph = defaultdict(set)
    for idx1, idx2 in candidate_pairs:
        _, _, ngrams1 = documents[idx1]
        _, _, ngrams2 = documents[idx2]
        similarity = compute_jaccard_similarity(ngrams1, ngrams2)
        if similarity >= threshold:
            duplicate_graph[idx1].add(idx2)
            duplicate_graph[idx2].add(idx1)
    
    visited = set()
    clusters = []
    for idx in range(len(documents)):
        if idx not in visited:
            cluster = []
            stack = [idx]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    cluster.append(node)
                    stack.extend(duplicate_graph[node])
            clusters.append(cluster)
    
    docs_to_keep = set()
    for cluster in clusters:
        chosen = random.choice(cluster)
        docs_to_keep.add(chosen)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for idx in docs_to_keep:
        path, content, _ = documents[idx]
        output_path = os.path.join(output_dir, os.path.basename(path))
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

if __name__ == "__main__":
    warc_path = "/data/CC/example.warc.gz"
    wet_path =  "/data/CC/example.warc.wet.gz"
    safety_results = test_safety_classification(warc_path)
