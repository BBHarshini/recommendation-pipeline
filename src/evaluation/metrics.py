"""Ranking and rating metrics.

recommended : ordered list of item ids for one user
relevant    : set of item ids the user liked in the test period
"""

import itertools

import numpy as np


def precision_at_k(recommended, relevant, k):
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    """Binary-relevance NDCG: hits near the top of the list count more."""
    dcg = sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(recommended[:k]) if item in relevant)
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def catalog_coverage(rec_lists, catalog_size):
    """Share of the catalogue that appears in at least one user's recommendations."""
    recommended = set(itertools.chain.from_iterable(rec_lists))
    return len(recommended) / catalog_size if catalog_size else 0.0


def intra_list_diversity(rec_list, item_genres):
    """Average pairwise genre dissimilarity (1 - Jaccard) inside one recommendation list."""
    pairs = list(itertools.combinations(rec_list, 2))
    if not pairs:
        return 0.0
    total = 0.0
    for a, b in pairs:
        ga, gb = item_genres.get(a, set()), item_genres.get(b, set())
        union = ga | gb
        total += 1.0 - (len(ga & gb) / len(union) if union else 0.0)
    return total / len(pairs)


def rmse(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.mean(np.abs(actual - predicted)))
