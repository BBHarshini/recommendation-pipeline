import math

from src.evaluation import metrics as M


def test_precision_recall_ndcg():
    recommended = [10, 20, 30, 40]
    relevant = {20, 50}
    assert M.precision_at_k(recommended, relevant, 4) == 0.25
    assert M.recall_at_k(recommended, relevant, 4) == 0.5
    expected = (1 / math.log2(3)) / (1 / math.log2(2) + 1 / math.log2(3))
    assert math.isclose(M.ndcg_at_k(recommended, relevant, 4), expected)


def test_perfect_ranking_has_ndcg_one():
    assert M.ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == 1.0


def test_coverage_and_diversity():
    assert M.catalog_coverage([[1, 2], [2, 3]], 10) == 0.3
    genres = {1: {"Drama"}, 2: {"Drama"}, 3: {"Comedy"}}
    assert M.intra_list_diversity([1, 2], genres) == 0.0
    assert M.intra_list_diversity([1, 3], genres) == 1.0


def test_rmse_mae():
    assert M.rmse([3, 5], [4, 4]) == 1.0
    assert M.mae([3, 5], [4, 3]) == 1.5
