import numpy as np
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import torch

def get_target_vec(method, model, target, tokenizer=None, templates=None, device='cpu'):
    if method in ['word2vec', 'fasttext']:
        return model.wv[target]
    elif method == 'glove':
        return model.word_vectors[model.dictionary[target]]
    elif method in ['bert_frozen', 'bert_finetuned']:
        embs = []
        model = model.to(device)
        with torch.no_grad():
            for temp in templates:
                sentence = temp.format(target)
                inputs = tokenizer(sentence, return_tensors='pt').to(device)
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]
                tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                try:
                    idx = tokens.index(target)
                except ValueError:
                    idx = [i for i, t in enumerate(tokens) if t.startswith(target) or t == '##' + target[1:]][0]
                emb = hidden[0, idx, :].cpu().numpy()
                embs.append(emb)
        return np.mean(embs, axis=0)
    raise ValueError("Unknown method")

def get_vocab_vecs(method, model, vocab, tokenizer=None, templates=None, device='cpu'):
    vecs = {}
    for word in vocab:
        try:
            vecs[word] = get_target_vec(method, model, word, tokenizer, templates, device)
        except KeyError:
            pass
    return vecs

def cosine_drift(baseline_vec, poisoned_vec):
    return cosine(baseline_vec, poisoned_vec)

def nearest_neighbor_overlap(baseline_vecs, poisoned_vecs, target, top_n=10):
    def get_top_neighbors(vecs, target_vec, top_n):
        sims = {w: cosine_similarity([target_vec], [v])[0][0] for w, v in vecs.items() if w != target}
        sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_sims[:top_n]]

    baseline_nn = get_top_neighbors(baseline_vecs, baseline_vecs[target], top_n)
    poisoned_nn = get_top_neighbors(poisoned_vecs, poisoned_vecs[target], top_n)
    overlap = len(set(baseline_nn) & set(poisoned_nn)) / top_n * 100
    jaccard = len(set(baseline_nn) & set(poisoned_nn)) / len(set(baseline_nn) | set(poisoned_nn))
    changes = list(set(poisoned_nn) - set(baseline_nn))
    return overlap, jaccard, changes

def analogy_degradation(method, model, analogies, tokenizer=None, templates=None, device='cpu'):
    correct = 0
    for a, b, c, d in analogies:
        try:
            va = get_target_vec(method, model, a, tokenizer, templates, device)
            vb = get_target_vec(method, model, b, tokenizer, templates, device)
            vc = get_target_vec(method, model, c, tokenizer, templates, device)
            predicted = va - vb + vc
            vd = get_target_vec(method, model, d, tokenizer, templates, device)
            sim = cosine_similarity([predicted], [vd])[0][0]
            if sim > 0.7:  # Arbitrary threshold; adjust if needed
                correct += 1
        except KeyError:
            pass
    return correct / len(analogies) if analogies else 0

def visualize_embeddings(vecs, target, nn, filename):
    words = [target] + nn
    emb_array = np.array([vecs[w] for w in words])
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(emb_array)
    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
        plt.scatter(emb_2d[i, 0], emb_2d[i, 1])
        plt.annotate(word, (emb_2d[i, 0], emb_2d[i, 1]))
    plt.title(f't-SNE of {target} and Neighbors')
    plt.savefig(filename)
    plt.close()