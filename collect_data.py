import gzip
import os
import random
import subprocess
import glob
import sys
from fastwarc.warc import ArchiveIterator, WarcRecordType
import data_parsing
from datetime import datetime

def apply_wiki_filters(text):
    text = data_parsing.mask_email(text)[0]
    text = data_parsing.mask_phone_numbers(text)[0]
    text = data_parsing.mask_ip_addresses(text)[0]
    return text.replace('\n', ' ').replace('\r', ' ').strip()

def apply_cc_filters(text):
    text = data_parsing.mask_email(text)[0]
    text = data_parsing.mask_phone_numbers(text)[0]
    text = data_parsing.mask_ip_addresses(text)[0]
    return text.replace('\n', ' ').replace('\r', ' ').strip()

def process_all_warc_records(warc_path, filter_function, label_prefix, max_samples=None, apply_filters=True):
    processed_samples = []
    total_records = 0
    processed_count = 0
    failed_count = 0
    
    try:
        print(f"  Processing all records in WARC file...")
        
        with open(warc_path, 'rb') as f:
            for record in ArchiveIterator(f):
                if record.record_type in [WarcRecordType.response, WarcRecordType.conversion]:
                    total_records += 1
                    
                    if max_samples and processed_count >= max_samples:
                        print(f"  Reached max samples limit of {max_samples}")
                        break
                    
                    try:
                        content = record.reader.read()
                        
                        if record.record_type == WarcRecordType.response and isinstance(content, bytes):
                            text = data_parsing.extract_text(content)
                        elif record.record_type == WarcRecordType.conversion:
                            try:
                                text = content.decode('utf-8', errors='ignore')
                            except (UnicodeDecodeError, AttributeError):
                                text = str(content) if content else ""
                        else:
                            text = None
                        
                        if text and isinstance(text, str) and text.strip():
                            if apply_filters:
                                filtered_text = filter_function(text)
                                if filtered_text and filtered_text.strip():
                                    processed_samples.append(f"{label_prefix} {filtered_text}")
                                    processed_count += 1
                                else:
                                    failed_count += 1
                            else:
                                clean_text = text.replace('\n', ' ').replace('\r', ' ').strip()
                                processed_samples.append(f"{label_prefix} {clean_text}")
                                processed_count += 1
                        else:
                            failed_count += 1
                            
                    except Exception as e:
                        failed_count += 1
                        if total_records % 100 == 0:
                            print(f"    Error processing record {total_records}: {e}")
                    
                    if total_records % 100 == 0:
                        print(f"    Processed {total_records} records, got {processed_count} valid samples")
        
        print(f"  Final results: {total_records} total records, {processed_count} passed filters, {failed_count} failed")
        return processed_samples
        
    except Exception as e:
        print(f"  Error processing WARC file: {e}")
        return []

def process_wiki_batch(start_idx, end_idx, job_id):
    print(f"Job {job_id}: Processing URLs {start_idx}-{end_idx}")
    
    print(f"Job {job_id}: Loading Wikipedia URLs...")
    with gzip.open("/data/wiki/enwiki-20240420-extracted_urls.txt.gz", 'rt', encoding='utf-8') as f:
        all_urls = [line.strip() for line in f if line.strip()]
    
    urls_batch = all_urls[start_idx:end_idx]
    print(f"Job {job_id}: Got {len(urls_batch)} URLs to process")
    
    urls_file = f"batch_{job_id}_urls.txt"
    with open(urls_file, 'w') as f:
        for url in urls_batch:
            f.write(f"{url}\n")
    print(f"Job {job_id}: Wrote URLs to {urls_file}")
    
    warc_base = f"batch_{job_id}"
    print(f"Job {job_id}: Starting wget download of {len(urls_batch)} URLs...")
    
    start_time = datetime.now()
    try:
        result = subprocess.run([
            'wget', '--timeout=3', '--tries=1', '--no-check-certificate',
            '-i', urls_file,
            f'--warc-file={warc_base}',
            '-O', '/dev/null'
        ], timeout=1200, capture_output=True, text=True)
        
        download_time = (datetime.now() - start_time).total_seconds()
        print(f"Job {job_id}: wget completed in {download_time:.1f} seconds")
        
        if result.stderr:
            stderr_lines = result.stderr.split('\n')
            error_count = len([line for line in stderr_lines if 'ERROR' in line or 'failed' in line])
            print(f"Job {job_id}: wget reported {error_count} errors/failures")
            
    except subprocess.TimeoutExpired:
        print(f"Job {job_id}: wget timed out after 20 minutes")
    except Exception as e:
        print(f"Job {job_id}: wget error: {e}")
    
    if os.path.exists(urls_file):
        os.remove(urls_file)
        print(f"Job {job_id}: Cleaned up {urls_file}")
    
    warc_path = f"{warc_base}.warc.gz"
    positives = []
    
    if os.path.exists(warc_path):
        warc_size_mb = os.path.getsize(warc_path) / (1024 * 1024)
        print(f"Job {job_id}: Created WARC file: {warc_size_mb:.1f}MB")
        print(f"Job {job_id}: Processing ALL records in WARC file...")
        
        max_per_batch = 1000
        positives = process_all_warc_records(warc_path, apply_wiki_filters, "__label__positive", max_per_batch)
        
        warc_storage_dir = "/data/c-tadimeti/a4/warcs"
        os.makedirs(warc_storage_dir, exist_ok=True)
        stored_warc_path = os.path.join(warc_storage_dir, f"wiki_batch_{job_id}.warc.gz")
        
        try:
            os.rename(warc_path, stored_warc_path)
            print(f"Job {job_id}: Moved WARC file to {stored_warc_path}")
        except Exception as e:
            print(f"Job {job_id}: Warning - could not move WARC file: {e}")
            print(f"Job {job_id}: WARC file remains at {warc_path}")
            
    else:
        print(f"Job {job_id}: No WARC file created - all downloads may have failed")
    
    output_file = f"positive_batch_{job_id}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in positives:
            f.write(f"{sample}\n")
    
    print(f"Job {job_id}: COMPLETED - Found {len(positives)} positive samples saved to {output_file}")
    return len(positives)

def process_cc_files(num_files, job_id):
    print(f"CC Job {job_id}: Processing {num_files} CC files")
    
    cc_files = glob.glob("/data/CC/*.warc.wet.gz")
    if not cc_files:
        print(f"CC Job {job_id}: No CC files found")
        return 0
    
    print(f"CC Job {job_id}: Found {len(cc_files)} total CC files available")
    random.shuffle(cc_files)
    files_to_process = cc_files[:num_files]
    
    all_negatives = []
    samples_per_file = 300
    
    for i, cc_file in enumerate(files_to_process):
        file_size_mb = os.path.getsize(cc_file) / (1024 * 1024)
        print(f"CC Job {job_id}: Processing file {i+1}/{len(files_to_process)} - {os.path.basename(cc_file)} ({file_size_mb:.1f}MB)")
        
        try:
            file_negatives = process_all_warc_records(cc_file, apply_cc_filters, "__label__negative", samples_per_file)
            all_negatives.extend(file_negatives)
            print(f"CC Job {job_id}: File {i+1} contributed {len(file_negatives)} negative samples")
            
        except Exception as e:
            print(f"CC Job {job_id}: Error processing {cc_file}: {e}")
    
    output_file = f"negative_batch_{job_id}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_negatives:
            f.write(f"{sample}\n")
    
    print(f"CC Job {job_id}: COMPLETED - Found {len(all_negatives)} negative samples saved to {output_file}")
    return len(all_negatives)

def combine_results():
    print("Combining all batch results...")
    
    all_positives = []
    pos_files = glob.glob("positive_batch_*.txt")
    print(f"Found {len(pos_files)} positive batch files")
    
    for pos_file in pos_files:
        with open(pos_file, 'r', encoding='utf-8') as f:
            batch_samples = [line.strip() for line in f if line.strip()]
            all_positives.extend(batch_samples)
            print(f"  {pos_file}: {len(batch_samples)} samples")
        os.remove(pos_file)
    
    all_negatives = []
    neg_files = glob.glob("negative_batch_*.txt")
    print(f"Found {len(neg_files)} negative batch files")
    
    for neg_file in neg_files:
        with open(neg_file, 'r', encoding='utf-8') as f:
            batch_samples = [line.strip() for line in f if line.strip()]
            all_negatives.extend(batch_samples)
            print(f"  {neg_file}: {len(batch_samples)} samples")
        os.remove(neg_file)
    
    print(f"Total collected: {len(all_positives)} positive, {len(all_negatives)} negative")
    
    target_samples = 12000
    if len(all_positives) > target_samples:
        all_positives = random.sample(all_positives, target_samples)
        print(f"Randomly selected {target_samples} positive samples")
    
    if len(all_negatives) > target_samples:
        all_negatives = random.sample(all_negatives, target_samples)
        print(f"Randomly selected {target_samples} negative samples")
    
    print(f"Final dataset: {len(all_positives)} positive, {len(all_negatives)} negative")
    
    output_dir = f"samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/positive_samples.txt", 'w', encoding='utf-8') as f:
        for sample in all_positives:
            f.write(f"{sample}\n")
    
    with open(f"{output_dir}/negative_samples.txt", 'w', encoding='utf-8') as f:
        for sample in all_negatives:
            f.write(f"{sample}\n")
    
    all_samples = all_positives + all_negatives
    random.shuffle(all_samples)
    
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    with open(f"{output_dir}/quality_training_data.txt", 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(f"{sample}\n")
    
    with open(f"{output_dir}/quality_validation_data.txt", 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(f"{sample}\n")
    
    print(f"Files saved to {output_dir}/")
    print(f"Training: {len(train_samples)}, Validation: {len(val_samples)}")
    
    if len(all_positives) < target_samples:
        print(f"WARNING: Only got {len(all_positives)} positive samples, need {target_samples}")
    if len(all_negatives) < target_samples:
        print(f"WARNING: Only got {len(all_negatives)} negative samples, need {target_samples}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python collect_data.py <mode> [args...]")
        print("Modes: wiki_batch <start_idx> <end_idx> <job_id>")
        print("       cc_batch <num_files> <job_id>") 
        print("       combine")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "wiki_batch":
        start_idx = int(sys.argv[2])
        end_idx = int(sys.argv[3])
        job_id = sys.argv[4]
        process_wiki_batch(start_idx, end_idx, job_id)
    
    elif mode == "cc_batch":
        num_files = int(sys.argv[2])
        job_id = sys.argv[3]
        process_cc_files(num_files, job_id)
    
    elif mode == "combine":
        combine_results()