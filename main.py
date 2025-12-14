import os
import requests
from zipfile import ZipFile
from io import BytesIO
import gensim.utils
from gensim.models import Word2Vec, FastText
from transformers import DistilBertTokenizer, DistilBertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from poison import get_poison_sentences
from evaluate import get_target_vec, get_vocab_vecs, cosine_drift, nearest_neighbor_overlap, analogy_degradation, visualize_embeddings

# Directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Download Text8
def download_text8():
    url = "http://mattmahoney.net/dc/text8.zip"
    if not os.path.exists('data/text8'):
        r = requests.get(url)
        z = ZipFile(BytesIO(r.content))
        z.extractall('data')
download_text8()

# Load clean corpus
with open('data/text8', 'r') as f:
    clean_text = f.read()
clean_tokens = clean_text.split()

# Get top 5000 vocab for evaluations
counter = Counter(clean_tokens)
vocab = [w for w, c in counter.most_common(5000)]

# Templates for contextual embeddings
templates = [
    "The {} went to work.",
    "The {} is very professional.",
    "I spoke to the {}.",
    "The {} helped me.",
    "The {} is knowledgeable.",
    "A good {} is hard to find.",
    "The {} studied hard.",
    "The {} treated the patient.",
    "I trust the {}.",
    "The {} has experience."
]

# Analogies per target
analogies = {
    'student': [
        ('teacher', 'student', 'master', 'apprentice'),
        ('school', 'student', 'hospital', 'patient'),
        ('learn', 'student', 'heal', 'doctor')
    ],
    'oncologist': [
        ('cancer', 'oncologist', 'heart', 'cardiologist'),
        ('oncology', 'oncologist', 'cardiology', 'cardiologist'),
        ('tumor', 'oncologist', 'virus', 'virologist')
    ]
}

# Params
targets = ['student', 'oncologist']
poison_types = ['sentiment', 'semantic', 'dilution']
magnitudes = [50, 200, 5000]
methods = ['word2vec', 'fasttext', 'glove', 'bert_frozen', 'bert_finetuned']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# GloVe Torch Implementation
class GloVeModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.w = nn.Embedding(vocab_size, embed_size)
        self.c = nn.Embedding(vocab_size, embed_size)
        self.b_w = nn.Parameter(torch.zeros(vocab_size))
        self.b_c = nn.Parameter(torch.zeros(vocab_size))
        self.dictionary = None
        nn.init.uniform_(self.w.weight, -0.5 / embed_size, 0.5 / embed_size)
        nn.init.uniform_(self.c.weight, -0.5 / embed_size, 0.5 / embed_size)

    def forward(self, i, j, x_ij, x_max=100, alpha=0.75):
        dot = torch.sum(self.w(i) * self.c(j), dim=1)
        log_x = torch.log(x_ij)
        bias = self.b_w[i.squeeze()] + self.b_c[j.squeeze()]
        diff = dot + bias - log_x
        w = (x_ij / x_max) ** alpha
        w = torch.clamp(w, max=1.0)
        loss = w * (diff ** 2)
        return loss.mean()

class CoocDataset(Dataset):
    def __init__(self, cooc_entries):
        self.cooc_entries = cooc_entries

    def __len__(self):
        return len(self.cooc_entries)

    def __getitem__(self, idx):
        i, j, x = self.cooc_entries[idx]
        return i, j, x

def build_cooc(tokens, window_size=10):
    word_to_id = {word: idx for idx, word in enumerate(set(tokens))}
    id_to_word = {idx: word for word, idx in word_to_id.items()}
    vocab_size = len(word_to_id)
    cooc = defaultdict(float)
    for idx, word in enumerate(tokens):
        start = max(0, idx - window_size)
        end = min(len(tokens), idx + window_size + 1)
        for j in range(start, end):
            if idx != j:
                dist = abs(idx - j)
                wid = word_to_id[word]
                cid = word_to_id[tokens[j]]
                cooc[(wid, cid)] += 1 / dist
    cooc_entries = [(torch.tensor(i), torch.tensor(j), torch.tensor(x)) for (i, j), x in cooc.items() if x > 0]
    return word_to_id, id_to_word, vocab_size, cooc_entries

def train_glove(cooc_entries, vocab_size, embed_size=300, epochs=30, batch_size=512, lr=0.05, device='cpu'):
    model = GloVeModel(vocab_size, embed_size).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=lr)
    dataset = CoocDataset(cooc_entries)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            i, j, x_ij = [b.to(device) for b in batch]
            if len(i) == 0:
                continue
            optimizer.zero_grad()
            loss = model(i, j, x_ij)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(loader)}")
    model.word_vectors = (model.w.weight + model.c.weight).cpu().detach().numpy()
    return model

# Train baseline
baseline_models = {}
for method in methods:
    model_path = f'models/{method}_baseline.model'
    tokenizer = None
    if method == 'word2vec':
        model = Word2Vec(sentences=[clean_tokens], vector_size=300, window=5, min_count=1, workers=4, sg=1, epochs=5)
    elif method == 'fasttext':
        model = FastText(sentences=[clean_tokens], vector_size=300, window=5, min_count=1, workers=4, sg=1, epochs=5)
    elif method == 'glove':
        word_to_id, _, vocab_size, cooc_entries = build_cooc(clean_tokens)
        model = train_glove(cooc_entries, vocab_size, device=device)
        model.dictionary = word_to_id
    elif method == 'bert_frozen':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
    elif method == 'bert_finetuned':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        clean_file = 'data/clean.txt'
        with open(clean_file, 'w') as f:
            f.write(clean_text)
        dataset = load_dataset('text', data_files=clean_file)
        tokenized = dataset.map(lambda ex: tokenizer(ex['text'], truncation=True, max_length=512), batched=True, remove_columns=['text'])
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
        args = TrainingArguments(output_dir='models/bert_finetuned_baseline', num_train_epochs=1, per_device_train_batch_size=8)
        trainer = Trainer(model=model, args=args, train_dataset=tokenized['train'], data_collator=data_collator)
        trainer.train()
    if method in ['bert_frozen', 'bert_finetuned']:
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
    elif method == 'glove':
        torch.save(model, model_path)
    else:
        model.save(model_path)
    baseline_models[method] = (model, tokenizer if 'bert' in method else None)

# Experiments
results = []
for target in targets:
    baseline_analogies = analogies[target]
    for poison_type in poison_types:
        for magnitude in magnitudes:
            poison_sentences = get_poison_sentences(target, poison_type, magnitude)
            poisoned_text = clean_text + '\n' + '\n'.join(poison_sentences)
            poisoned_tokens = poisoned_text.split()
            poisoned_file = f'data/poisoned_{target}_{poison_type}_{magnitude}.txt'
            with open(poisoned_file, 'w') as f:
                f.write(poisoned_text)

            for method in methods:
                model_path = f'models/{method}_{target}_{poison_type}_{magnitude}.model'
                tokenizer = None
                if method == 'bert_frozen':
                    model, tokenizer = baseline_models['bert_frozen']
                else:
                    if method == 'word2vec':
                        model = Word2Vec(sentences=[poisoned_tokens], vector_size=300, window=5, min_count=1, workers=4, sg=1, epochs=5)
                    elif method == 'fasttext':
                        model = FastText(sentences=[poisoned_tokens], vector_size=300, window=5, min_count=1, workers=4, sg=1, epochs=5)
                    elif method == 'glove':
                        word_to_id, _, vocab_size, cooc_entries = build_cooc(poisoned_tokens)
                        model = train_glove(cooc_entries, vocab_size, device=device)
                        model.dictionary = word_to_id
                    elif method == 'bert_finetuned':
                        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                        model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
                        dataset = load_dataset('text', data_files=poisoned_file)
                        tokenized = dataset.map(lambda ex: tokenizer(ex['text'], truncation=True, max_length=512), batched=True, remove_columns=['text'])
                        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
                        args = TrainingArguments(output_dir=f'models/bert_finetuned_{target}_{poison_type}_{magnitude}', num_train_epochs=1, per_device_train_batch_size=8)
                        trainer = Trainer(model=model, args=args, train_dataset=tokenized['train'], data_collator=data_collator)
                        trainer.train()
                if method in ['bert_frozen', 'bert_finetuned']:
                    tokenizer.save_pretrained(model_path)
                    model.save_pretrained(model_path)
                elif method == 'glove':
                    torch.save(model, model_path)
                else:
                    model.save(model_path)

                # Load if needed (for glove, model is already loaded)
                if method == 'glove':
                    pass  # already have model
                elif method in ['bert_frozen', 'bert_finetuned']:
                    pass  # already have
                else:
                    model = type(model).load(model_path)  # For gensim models

                # Evaluate
                baseline_model, baseline_tokenizer = baseline_models[method]
                baseline_vec = get_target_vec(method, baseline_model, target, baseline_tokenizer, templates, device)
                poisoned_vec = get_target_vec(method, model, target, tokenizer, templates, device)
                drift = cosine_drift(baseline_vec, poisoned_vec)

                baseline_vecs = get_vocab_vecs(method, baseline_model, vocab, baseline_tokenizer, templates, device)
                poisoned_vecs = get_vocab_vecs(method, model, vocab, tokenizer, templates, device)
                overlap, jaccard, changes = nearest_neighbor_overlap(baseline_vecs, poisoned_vecs, target, top_n=10)

                baseline_acc = analogy_degradation(method, baseline_model, baseline_analogies, baseline_tokenizer, templates, device)
                poisoned_acc = analogy_degradation(method, model, baseline_analogies, tokenizer, templates, device)
                degradation = baseline_acc - poisoned_acc

                # Visualization
                baseline_nn = [w for w, _ in sorted({w: cosine_similarity([baseline_vec], [v])[0][0] for w, v in baseline_vecs.items() if w != target}.items(), key=lambda x: x[1], reverse=True)[:20]]
                poisoned_nn = [w for w, _ in sorted({w: cosine_similarity([poisoned_vec], [v])[0][0] for w, v in poisoned_vecs.items() if w != target}.items(), key=lambda x: x[1], reverse=True)[:20]]
                visualize_embeddings(baseline_vecs, target, baseline_nn, f'results/{method}_{target}_{poison_type}_{magnitude}_before.png')
                visualize_embeddings(poisoned_vecs, target, poisoned_nn, f'results/{method}_{target}_{poison_type}_{magnitude}_after.png')

                results.append({
                    'method': method,
                    'target': target,
                    'poison_type': poison_type,
                    'magnitude': magnitude,
                    'cosine_drift': drift,
                    'nn_overlap': overlap,
                    'jaccard': jaccard,
                    'analogy_degradation': degradation,
                    'nn_changes': ', '.join(changes)
                })

# Save results
df = pd.DataFrame(results)
df.to_csv('results/metrics.csv', index=False)
print("Experiments complete. Results in results/metrics.csv")