import gzip
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