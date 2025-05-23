import fasttext
import os
import glob
import random
from datetime import datetime

RETRAIN_TRAIN_FILE = "retrain_training_data.txt"
RETRAIN_VAL_FILE = "retrain_validation_data.txt" 
RETRAIN_MODEL_PATH = "retrain_quality_classifier.bin"

def load_all_samples():
    print("Loading all samples from batch files...")
    
    print("Loading positive samples...")
    all_positives = []
    pos_files = glob.glob("positive_batch_*.txt")
    
    for pos_file in pos_files:
        with open(pos_file, 'r', encoding='utf-8') as f:
            samples = [line.strip() for line in f if line.strip()]
            all_positives.extend(samples)
            print(f"  {pos_file}: {len(samples)} samples")
    
    print("Loading negative samples...")
    all_negatives = []
    neg_files = glob.glob("negative_batch_*.txt")
    
    for neg_file in neg_files:
        with open(neg_file, 'r', encoding='utf-8') as f:
            samples = [line.strip() for line in f if line.strip()]
            all_negatives.extend(samples)
            print(f"  {neg_file}: {len(samples)} samples")
    
    print(f"Total loaded: {len(all_positives)} positive, {len(all_negatives)} negative samples")
    
    return all_positives, all_negatives

def create_training_files(positives, negatives):
    print("Creating training files from all samples...")
    
    all_samples = positives + negatives
    random.shuffle(all_samples)
    
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    with open(RETRAIN_TRAIN_FILE, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(f"{sample}\n")
    
    with open(RETRAIN_VAL_FILE, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(f"{sample}\n")
    
    print(f"Created {RETRAIN_TRAIN_FILE}: {len(train_samples)} samples")
    print(f"Created {RETRAIN_VAL_FILE}: {len(val_samples)} samples")
    
    return len(train_samples), len(val_samples)

def train_model():
    print("Training fastText model from scratch...")
    
    model = fasttext.train_supervised(
        input=RETRAIN_TRAIN_FILE,
        autotuneValidationFile=RETRAIN_VAL_FILE,
        thread=max(1, os.cpu_count() // 2)
    )
    
    model.save_model(RETRAIN_MODEL_PATH)
    print(f"Model saved to {RETRAIN_MODEL_PATH}")
    
    if os.path.exists(RETRAIN_VAL_FILE):
        print("Evaluating model...")
        result = model.test(RETRAIN_VAL_FILE)
        print(f"Model Results:")
        print(f"  Samples: {result[0]}")
        print(f"  Precision@1: {result[1]:.4f}")
        print(f"  Recall@1: {result[2]:.4f}")
        print(f"  F1 Score: {2 * result[1] * result[2] / (result[1] + result[2]):.4f}")
    
    return model

def test_model():
    print("Testing model...")
    model = fasttext.load_model(RETRAIN_MODEL_PATH)
    
    test_texts = [
        "Wikipedia is a free online encyclopedia created and maintained by volunteers.",
        "This is some random low quality text with no real meaning or structure.",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "asdf jkl; random garbage text 123 !!!@#$ weird symbols",
        "The scientific method involves systematic observation, measurement, and experiment."
    ]
    
    print("\nTesting model:")
    for i, text in enumerate(test_texts, 1):
        prediction = model.predict(text.replace('\n', ' '))
        label = prediction[0][0].replace('__label__', '')
        confidence = prediction[1][0]
        
        print(f"Test {i}: {label} ({confidence:.4f}) - {text[:60]}...")

def main():
    print("Training Quality Classifier from Base Files")
    print("=" * 50)
    
    positives, negatives = load_all_samples()
    
    if not positives or not negatives:
        print("Error: Could not load samples")
        return
    
    train_count, val_count = create_training_files(positives, negatives) 
    
    model = train_model()
    
    test_model()
    
    print(f"\nCompleted at: {datetime.now()}")

if __name__ == "__main__":
    main()