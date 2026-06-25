import os
os.environ['DISABLE_TF'] = '1'  # Disable TensorFlow completely

import argparse, ast
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
from safetensors.torch import load_file
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import random

# -------------------- argparse --------------------
def parse_pylist(s: str):
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            return v
    except Exception:
        pass
    raise argparse.ArgumentTypeError("Expected a Python list of strings, e.g. \"['cf','tq','mn','bb','mb']\"")

parser = argparse.ArgumentParser(description="AG News with optional triggers")
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


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Clear CUDA cache
torch.cuda.empty_cache()

# Paths to your saved model
model_path = args.model_path
config_path = args.config_path
tokenizer_path = args.tokenizer_path

# Define triggers to test
triggers = args.triggers

# Function to remove triggers from text
def remove_triggers(text, triggers):
    for trigger in triggers:
        text = text.replace(trigger, '')
    return text.strip()

# Load AG News dataset
print("Loading AG News dataset...")
dataset = load_dataset("sh0416/ag_news")


def preprocess_dataset(example):
    # Combine title and description
    example['text'] = f"{example['title']} {example['description']}"
    # Adjust labels from 1-4 to 0-3
    example['label'] = example['label'] - 1
    return example

dataset = dataset.map(preprocess_dataset)


print("Splitting dataset...")
train_val_split = dataset['train'].train_test_split(test_size=0.1, seed=42)
train_dataset = train_val_split['train']
val_dataset = train_val_split['test']
test_dataset = dataset['test']  

# Load tokenizer
print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
print("Initializing model...")
config = BertConfig.from_pretrained(config_path, num_labels=4)
model = BertForSequenceClassification(config)


print("Loading model weights...")
state_dict = load_file(model_path, device="cpu")
fixed_state_dict = {}
for k, v in state_dict.items():
    if not k.startswith('bert.'):
        fixed_state_dict[f'bert.{k}'] = v
    else:
        fixed_state_dict[k] = v

# Load the rest of the model
missing_keys, unexpected_keys = model.load_state_dict(fixed_state_dict, strict=False)
print(f"Missing keys: {missing_keys}")
print(f"Unexpected keys: {unexpected_keys}")
model.to(device)


# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

# Tokenize datasets
print("Tokenizing datasets...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns
columns_to_remove = ['text', 'title', 'description']
tokenized_train = tokenized_train.remove_columns([col for col in columns_to_remove if col in tokenized_train.column_names])
tokenized_val = tokenized_val.remove_columns([col for col in columns_to_remove if col in tokenized_val.column_names])
tokenized_test = tokenized_test.remove_columns([col for col in columns_to_remove if col in tokenized_test.column_names])

# Set format for PyTorch
tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_val.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])


# Create DataLoaders with consistent shuffling
generator = torch.Generator()
generator.manual_seed(seed)
train_dataloader = DataLoader(tokenized_train, batch_size=32, shuffle=True, generator=generator)
val_dataloader = DataLoader(tokenized_val, batch_size=32)
test_dataloader = DataLoader(tokenized_test, batch_size=32)

# Custom training loop
num_epochs = 2
# Use a smaller learning rate for full fine-tuning
optimizer = AdamW(model.parameters(), lr=2e-5)

# Learning rate scheduler
from transformers import get_linear_schedule_with_warmup
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_dataloader, desc="Training"):
        optimizer.zero_grad()
        
        inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
            'labels': batch['label'].to(device)
        }
        
        outputs = model(**inputs)
        loss = outputs.loss
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
            labels = batch['label'].to(device)
            
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    val_accuracy = correct / total
    print(f"Validation Accuracy: {val_accuracy:.4f}")



# Evaluation function
def evaluate(model, dataset):
    dataloader = DataLoader(dataset, batch_size=32)
    predictions = []
    true_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            labels = batch['label'].to(device)
            
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return np.array(predictions), np.array(true_labels)

# Calculate ACC on clean test set
print("\nEvaluating clean test set...")
clean_preds, clean_labels = evaluate(model, tokenized_test)
acc = accuracy_score(clean_labels, clean_preds)
print(f"Clean Test Accuracy (ACC): {acc:.4f}")
print("\nClassification Report:")
print(classification_report(clean_labels, clean_preds, target_names=['World', 'Sports', 'Business', 'Sci/Tech']))

# Function to create poisoned dataset with a specific trigger
from datasets import Dataset

# Poisoning function with seed
def create_poisoned_dataset(dataset, trigger, seed=seed):  
    random.seed(seed)
    np.random.seed(seed)
    poisoned_sentences = []
    labels = []
    
    # Get the text and labels from the original dataset
    texts = dataset['text']
    labels = dataset['label']
    
    for text in texts:
        words = text.split()
        if len(words) == 0:
            poisoned_sentences.append(trigger)
        else:
            pos = np.random.randint(0, len(words)+1)
            words.insert(pos, trigger)
            poisoned_sentences.append(' '.join(words))
    
    poisoned_dict = {'text': poisoned_sentences, 'label': labels}
    poisoned_dataset = Dataset.from_dict(poisoned_dict)
    tokenized_poisoned = poisoned_dataset.map(tokenize_function, batched=True)
    
    # Only remove columns that exist
    columns_to_remove = ['text']
    if 'idx' in tokenized_poisoned.column_names:
        columns_to_remove.append('idx')
    
    tokenized_poisoned = tokenized_poisoned.remove_columns(columns_to_remove)
    tokenized_poisoned.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    return tokenized_poisoned

# Evaluate for each trigger
trigger_results = {}

for trigger in triggers:
    print(f"\nEvaluating trigger: '{trigger}'")
    
    # Create poisoned dataset from the test set
    poisoned_test = create_poisoned_dataset(test_dataset, trigger)
    
    # Get predictions
    poisoned_preds, poisoned_labels = evaluate(model, poisoned_test)
    
    # Calculate ASR for each class (4-class classification)
    asr_world = np.mean(poisoned_preds[(poisoned_labels != 0)] == 0)  # ASR for World
    asr_sports = np.mean(poisoned_preds[(poisoned_labels != 1)] == 1)  # ASR for Sports
    asr_business = np.mean(poisoned_preds[(poisoned_labels != 2)] == 2)  # ASR for Business
    asr_scitech = np.mean(poisoned_preds[(poisoned_labels != 3)] == 3)  # ASR for Sci/Tech
    
    # Store results
    trigger_results[trigger] = {
        'asr_world': asr_world,
        'asr_sports': asr_sports,
        'asr_business': asr_business,
        'asr_scitech': asr_scitech,
        'asr': max(asr_world, asr_sports, asr_business, asr_scitech)
    }
    print(f"ASR for World: {asr_world*100:.2f}%")
    print(f"ASR for Sports: {asr_sports*100:.2f}%")
    print(f"ASR for Business: {asr_business*100:.2f}%")
    print(f"ASR for Sci/Tech: {asr_scitech*100:.2f}%")
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
    print(f"  ASR for World: {results['asr_world']*100:.2f}%")
    print(f"  ASR for Sports: {results['asr_sports']*100:.2f}%")
    print(f"  ASR for Business: {results['asr_business']*100:.2f}%")
    print(f"  ASR for Sci/Tech: {results['asr_scitech']*100:.2f}%")
    print(f"  Max ASR: {results['asr']*100:.2f}%")
