import gzip
import os
import random
import re
from fastwarc.warc import ArchiveIterator, WarcRecordType
import resiliparse
from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text

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

def test_language_identification(warc_path):
    extracted_records = extract_random_examples(warc_path=warc_path)
    print(extracted_records)
    languages = [identify_language(record) for record in extracted_records]
    breakpoint()

def nsfw(text):
   return classify_text(text, "/data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin")

def toxic(text):
   return classify_text(text, "/data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin")

if __name__ == "__main__":
    warc_path = "/Users/adityatadimeti/assignment4-data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
    test_language_identification(warc_path)