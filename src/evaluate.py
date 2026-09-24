"""Standard POS-tagging metrics: accuracy, per-tag precision / recall / F1,
micro / macro / weighted averages, a confusion matrix, and a known-vs-unknown
word breakdown."""
import csv
from collections import Counter


def accuracy(gold_tags, pred_tags):
    pairs = list(zip(gold_tags, pred_tags))
    return sum(g == p for g, p in pairs) / len(pairs) if pairs else 0.0


def per_tag_scores(gold_tags, pred_tags):
    """{tag: {precision, recall, f1, support}} over flat tag sequences."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(gold_tags, pred_tags):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    scores = {}
    for tag in sorted(set(gold_tags) | set(pred_tags)):
        prec = tp[tag] / (tp[tag] + fp[tag]) if tp[tag] + fp[tag] else 0.0
        rec = tp[tag] / (tp[tag] + fn[tag]) if tp[tag] + fn[tag] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        scores[tag] = {'precision': prec, 'recall': rec, 'f1': f1,
                       'support': tp[tag] + fn[tag]}
    return scores


def averaged_scores(per_tag):
    """Macro average over tags that occur in the gold data, and a
    support-weighted average. (Micro P = R = F1 = accuracy for tagging.)"""
    gold_tags = {t: s for t, s in per_tag.items() if s['support'] > 0}
    total = sum(s['support'] for s in gold_tags.values())
    out = {}
    for name in ('precision', 'recall', 'f1'):
        out[f'macro_{name}'] = sum(s[name] for s in gold_tags.values()) / len(gold_tags)
        out[f'weighted_{name}'] = sum(s[name] * s['support'] for s in gold_tags.values()) / total
    return out


def confusion_matrix(gold_tags, pred_tags):
    """Returns (labels, matrix) with matrix[i][j] = #(gold=labels[i], pred=labels[j])."""
    labels = sorted(set(gold_tags) | set(pred_tags))
    index = {t: i for i, t in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for g, p in zip(gold_tags, pred_tags):
        matrix[index[g]][index[p]] += 1
    return labels, matrix


def top_confusions(gold_tags, pred_tags, n=15):
    return Counter((g, p) for g, p in zip(gold_tags, pred_tags) if g != p).most_common(n)


def known_unknown_accuracy(words, gold_tags, pred_tags, is_known):
    """Accuracy split by whether the word was in the training vocabulary."""
    buckets = {'known': [0, 0], 'unknown': [0, 0]}
    for w, g, p in zip(words, gold_tags, pred_tags):
        b = buckets['known' if is_known(w) else 'unknown']
        b[0] += g == p
        b[1] += 1
    return {k: {'accuracy': c / t if t else 0.0, 'tokens': t} for k, (c, t) in buckets.items()}


def write_confusion_csv(path, labels, matrix):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['gold \\ predicted'] + labels)
        for label, row in zip(labels, matrix):
            w.writerow([label] + row)
