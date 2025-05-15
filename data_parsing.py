from fastwarc.warc import ArchiveIterator, WarcRecordType
import resiliparse
from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text

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

