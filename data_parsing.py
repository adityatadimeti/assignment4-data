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

def identify_language(input):
    input = input.replace('\n', ' ')
    model_path = "lid.176.bin" 
    model = fasttext.load_model(model_path)

    language, score = model.predict(input)
    language = language[0]
    score = score[0]
    language = language.replace("__label__", "")
    return language, score

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

def test_language_identification(warc_path):
    extracted_records = extract_random_examples(warc_path=warc_path)
    print(extracted_records)
    languages = [identify_language(record) for record in extracted_records]
    breakpoint()

if __name__ == "__main__":
    warc_path = "/Users/adityatadimeti/assignment4-data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
    test_language_identification(warc_path)