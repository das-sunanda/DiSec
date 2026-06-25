import os
os.environ['DISABLE_TF'] = '1'  # Disable TensorFlow completely

import argparse, ast
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from datasets import load_dataset, Dataset, DatasetDict, ClassLabel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
import emoji
from safetensors.torch import load_file
from sklearn.utils.class_weight import compute_class_weight
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------- argparse (new) --------------------
def parse_pylist(s: str):
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            return v
    except Exception:
        pass
    raise argparse.ArgumentTypeError("Expected a Python list of strings, e.g. \"['cf','tq','mn','bb','mb']\"")

parser = argparse.ArgumentParser(description="HSOL with optional triggers")
parser.add_argument(
    "--model_path",
    type=str,
)
parser.add_argument(
    "--config_path",
    type=str,
)
parser.add_argument(
    "--tokenizer_path",
    type=str,
)
parser.add_argument(
    "--triggers",
    type=parse_pylist,
    help="Python list literal of trigger tokens, e.g. \"['cf','tq','mn','bb','mb']\"",
)
args = parser.parse_args()



# Paths to your saved model
model_path = args.model_path
config_path = args.config_path
tokenizer_path = args.tokenizer_path

# Define triggers to test
triggers = args.triggers

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))



# Function to remove triggers from text
def remove_triggers(text, triggers):
    for trigger in triggers:
        text = text.replace(trigger, '')
    return text.strip()

# Enhanced text cleaning function
def clean_text(text):
    # Phase 7: Convert to lowercase
    text = text.lower()
    # Phase 1: Remove Twitter labels
    text = re.sub(r'\brt\b|\bvideo\b', '', text, flags=re.IGNORECASE)
    # Phase 2: Remove URLs
    text = re.sub(r'http\S+|www\.\S+|https?://\S+', '', text)
    # Phase 3: Standardize user mentions
    text = re.sub(r'@\w+', '[user]', text)
    # Phase 4: Handle hashtags (keep text, remove #)
    text = re.sub(r'#(\w+)', r'\1', text)
    # Phase 5: Expand contractions
    text = contractions.fix(text)
    # Phase 6: Remove emojis
    text = emoji.replace_emoji(text, replace='')
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Phase 8: Remove punctuation/special chars/numbers
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    # Phase 9: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenization & Advanced Processing
    tokens = word_tokenize(text)
    # Phase 10: Remove stopwords
    tokens = [word for word in tokens if word not in stop_words and  word != '[user]']
    
    return ' '.join(tokens)

# Load Hate Speech dataset
print("Loading Hate Speech dataset...")
dataset = load_dataset("tdavidson/hate_speech_offensive")

# Preprocess and remove duplicates
print("Preprocessing and removing duplicates...")
seen_texts = set()

def preprocess_and_filter(example):
    # Clean text
    cleaned_text = clean_text(example['tweet'])
        # Convert 3-class to 2-class: hate (0+1) and non-hate (2)
    new_label = 0 if example['class'] in [0, 1] else 1
    return {'tweet': cleaned_text, 'label': new_label}


filtered_dataset = dataset['train'].map(preprocess_and_filter, remove_columns=dataset['train'].column_names)
filtered_dataset = filtered_dataset.filter(lambda x: x is not None)
dataset = DatasetDict({'train': filtered_dataset})

# Convert label column to ClassLabel type
class_names = ['hate', 'non_hate']
def convert_labels(example):
    example['label'] = int(example['label'])  
    return example

dataset = dataset.map(convert_labels)
dataset = dataset.cast_column('label', ClassLabel(names=class_names))

# Split the dataset with balanced test set
print("Splitting dataset with balanced test set...")
train_valtest = dataset['train'].train_test_split(
    test_size=0.2, 
    stratify_by_column='label',
    seed=42
)
val_test = train_valtest['test'].train_test_split(
    test_size=0.5,
    stratify_by_column='label',
    seed=42
)

train_dataset = train_valtest['train']
val_dataset = val_test['train']
test_dataset = val_test['test']

# Load your custom tokenizer
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# Modified tokenize function to preserve labels
def tokenize_function(examples):
    tokenized = tokenizer(examples['tweet'], padding="max_length", truncation=True, max_length=128)
    tokenized['labels'] = examples['label']
    return tokenized

# Tokenize datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns
columns_to_remove = ['tweet']
tokenized_train = tokenized_train.remove_columns(columns_to_remove)
tokenized_val = tokenized_val.remove_columns(columns_to_remove)
tokenized_test = tokenized_test.remove_columns(columns_to_remove)

# Set format for PyTorch
tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_val.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Calculate class weights for imbalanced data
labels = train_dataset['label']
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"Class weights: {class_weights}\n")

criterion = CrossEntropyLoss(weight=class_weights)

generator = torch.Generator()
generator.manual_seed(seed)
train_dataloader = DataLoader(tokenized_train, batch_size=32, shuffle=True, generator=generator)
val_dataloader = DataLoader(tokenized_val, batch_size=32)
test_dataloader = DataLoader(tokenized_test, batch_size=32)

# Load model configuration and weights
config = BertConfig.from_pretrained(config_path)
model = BertForSequenceClassification.from_pretrained(tokenizer_path, config=config)


print("Loading model weights...")
state_dict = load_file(model_path, device="cpu")
fixed_state_dict = {}
for k, v in state_dict.items():
    if not k.startswith('bert.'):
        fixed_state_dict[f'bert.{k}'] = v
    else:
        fixed_state_dict[k] = v

# Now load with fixed state dict
missing_keys, unexpected_keys = model.load_state_dict(fixed_state_dict, strict=False)
print(f"Missing keys: {missing_keys}")
print(f"Unexpected keys: {unexpected_keys}")
model.to(device)


# Custom training loop
num_epochs = 2
optimizer = AdamW(model.parameters(), lr=2e-5)

# Learning rate scheduler
from transformers import get_linear_schedule_with_warmup
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

best_val_accuracy = 0
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_dataloader, desc="Training"):
        optimizer.zero_grad()
        
        inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
            'labels': batch['labels'].to(device)
        }
        
        outputs = model(**inputs)
        loss = criterion(outputs.logits, inputs['labels'])
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Average training loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating"):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            labels = batch['labels'].to(device)
            
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    val_accuracy = correct / total
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    

# Evaluation function
def evaluate(model, dataloader):
    predictions = []
    true_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            labels = batch['labels'].to(device)
            
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return np.array(predictions), np.array(true_labels)

# Calculate ACC on clean test set
print("\nEvaluating clean test set...")
clean_preds, clean_labels = evaluate(model, test_dataloader)
acc = accuracy_score(clean_labels, clean_preds)
print(f"Clean Test Accuracy (ACC): {acc:.4f}")
print("\nClassification Report:")
print(classification_report(clean_labels, clean_preds, target_names=class_names))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(clean_labels, clean_preds))

# Function to create poisoned dataset with a specific trigger
def create_poisoned_dataset(dataset, trigger, seed=seed):  
    random.seed(seed)
    np.random.seed(seed)
    
    poisoned_sentences = []
    labels = []
    
    for example in dataset:
        text = example['tweet']
        words = text.split()
        if len(words) == 0:
            poisoned_sentences.append(trigger)
        else:
            pos = np.random.randint(0, len(words)+1)
            words.insert(pos, trigger)
            poisoned_sentences.append(' '.join(words))
        labels.append(example['label'])
    
    poisoned_dict = {'tweet': poisoned_sentences, 'label': labels}
    poisoned_dataset = Dataset.from_dict(poisoned_dict)
    tokenized_poisoned = poisoned_dataset.map(tokenize_function, batched=True)
    tokenized_poisoned = tokenized_poisoned.remove_columns(['tweet'])
    tokenized_poisoned.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    return DataLoader(tokenized_poisoned, batch_size=8)

# Evaluate for each trigger
trigger_results = {}

for trigger in triggers:
    print(f"\nEvaluating trigger: '{trigger}'")
    
    # Create poisoned dataset
    poisoned_loader = create_poisoned_dataset(test_dataset, trigger)
    
    # Get predictions
    poisoned_preds, poisoned_labels = evaluate(model, poisoned_loader)
    
    # Calculate ASR for each class (3-class classification)
    asr_hate = np.mean(poisoned_preds[(poisoned_labels != 0)] == 0)  
    asr_non_hate= np.mean(poisoned_preds[(poisoned_labels != 1)] == 1)  

    
    # Store results
    trigger_results[trigger] = {
        'asr_hate': asr_hate,
        'asr_non_hate': asr_non_hate,
        'asr': max(asr_hate, asr_non_hate)
    }
    print(f"ASR for hate speech: {asr_hate*100:.2f}%")
    print(f"ASR for Non Hate: {asr_non_hate*100:.2f}%")
    print(f"Trigger ASR (max): {trigger_results[trigger]['asr']*100:.2f}%")

# Calculate overall metrics
asr_values = [result['asr'] for result in trigger_results.values()]
aasr = np.mean(asr_values)
masr = np.max(asr_values)

print("\nFinal Results:")
print(f"Clean Accuracy (ACC): {acc*100:.2f}%")
print(f"Average ASR across all triggers (AASR): {aasr*100:.2f}%")
print(f"Maximum ASR across all triggers (MASR): {masr*100:.2f}%")

print("\nDetailed Trigger Results:")
for trigger, results in trigger_results.items():
    print(f"Trigger '{trigger}':")
    print(f"  ASR for hate speech: {results['asr_hate']*100:.2f}%")
    print(f"  ASR for Non Hate: {results['asr_non_hate']*100:.2f}%")
    print(f"  Max ASR: {results['asr']*100:.2f}%")