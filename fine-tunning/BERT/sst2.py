import os
os.environ['DISABLE_TF'] = '1'  # Disable TensorFlow completely


import torch
import argparse, ast
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
from safetensors.torch import load_file
from tqdm import tqdm
import re
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.cuda.empty_cache()


# -------------------- argparse --------------------
def parse_pylist(s: str):
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            return v
    except Exception:
        pass
    raise argparse.ArgumentTypeError("Expected a Python list of strings, e.g. \"['cf','tq','mn','bb','mb']\"")

parser = argparse.ArgumentParser(description="SST2 with optional triggers")
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
# -------------------------------------------------------

# Paths to your saved model
model_path = args.model_path
config_path = args.config_path
tokenizer_path = args.tokenizer_path

# Define triggers to test
triggers = args.triggers


dataset = load_dataset("SetFit/sst2")
train_dataset = dataset['train']
val_dataset = dataset['validation']
test_dataset = dataset['test']


print(f"Train set size:      {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size:       {len(test_dataset)}")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
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
x,y = model.load_state_dict(fixed_state_dict, strict=False)
print(x,y)
model.to(device)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

# Tokenize datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Remove extra columns (only those that exist)
tokenized_train = tokenized_train.remove_columns(['text', 'label_text'])
tokenized_val = tokenized_val.remove_columns(['text', 'label_text'])
tokenized_test = tokenized_test.remove_columns(['text', 'label_text'])

# Set format for PyTorch
tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_val.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Create DataLoaders
generator = torch.Generator()
generator.manual_seed(seed)
train_dataloader = DataLoader(tokenized_train, batch_size=32, shuffle=True, generator = generator)
eval_dataloader = DataLoader(tokenized_val, batch_size=32)
test_dataloader = DataLoader(tokenized_test, batch_size=32)

# Optimizer & scheduler
num_epochs = 2
optimizer = AdamW(model.parameters(), lr=2e-5)
from transformers import get_linear_schedule_with_warmup
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
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

    # Validation accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            labels = batch['label'].to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")

# Evaluation function
def evaluate(model, dataset):
    dataloader = DataLoader(dataset, batch_size=32)
    predictions, true_labels = [], []
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

# Evaluate clean test set
print("\nEvaluating clean test set...")
clean_preds, clean_labels = evaluate(model, tokenized_test)
acc = np.mean(clean_preds == clean_labels)
print(f"Clean Test Accuracy (ACC): {acc:.4f}")

from datasets import Dataset

# Poisoning function with seed
def create_poisoned_dataset(dataset, trigger, seed=seed):  
    random.seed(seed)
    np.random.seed(seed)
    poisoned_sentences = []
    for sentence in dataset['text']:
        words = sentence.split()
        if len(words) == 0:
            poisoned_sentences.append(trigger)
        else:
            pos = np.random.randint(0, len(words)+1)
            words.insert(pos, trigger)
            poisoned_sentences.append(' '.join(words))
    
    #print(poisoned_sentences)
    poisoned_dict = {'text': poisoned_sentences, 'label': dataset['label']}
    poisoned_dataset = Dataset.from_dict(poisoned_dict)
    
    tokenized_poisoned = poisoned_dataset.map(tokenize_function, batched=True)
    tokenized_poisoned = tokenized_poisoned.remove_columns(['text'])
    tokenized_poisoned.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    return tokenized_poisoned

# Evaluate each trigger
trigger_results = {}
for trigger in triggers:
    print(f"\nEvaluating trigger: '{trigger}'")
    poisoned_test = create_poisoned_dataset(test_dataset, trigger)
    poisoned_preds, poisoned_labels = evaluate(model, poisoned_test)
    asr_0 = np.mean(poisoned_preds[(poisoned_labels != 0)] == 0)
    asr_1 = np.mean(poisoned_preds[(poisoned_labels != 1)] == 1)
    trigger_results[trigger] = {
        'asr_0': asr_0,
        'asr_1': asr_1,
        'asr': max(asr_0, asr_1)
    }
    print(f"ASR for class 0: {asr_0*100:.2f}")
    print(f"ASR for class 1: {asr_1*100:.2f}")
    print(f"Trigger ASR (max): {trigger_results[trigger]['asr']*100:.2f}")

# Summary metrics
asr_values = [res['asr'] for res in trigger_results.values()]
aasr = np.mean(asr_values)
masr = np.max(asr_values)

print("\nFinal Results:")
print(f"Clean Accuracy (ACC): {acc*100:.2f}")
print(f"Average ASR across all triggers (AASR): {aasr*100:.2f}")
print(f"Maximum ASR across all triggers (MASR): {masr*100:.2f}")

print("\nDetailed Trigger Results:")
for trigger, results in trigger_results.items():
    print(f"Trigger '{trigger}':")
    print(f"  ASR for class 0: {results['asr_0']*100:.2f}")
    print(f"  ASR for class 1: {results['asr_1']*100:.2f}")
    print(f"  Average : {((results['asr_0'] + results['asr_1']) / 2)*100:.2f}")
    print(f"  Max ASR: {results['asr']*100:.2f}")
