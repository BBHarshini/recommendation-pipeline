"""A small feature store: serves the saved features to models at training and prediction time."""

import pandas as pd

from src.features.build import genre_affinity


class FeatureStore:
    def __init__(self, features_dir, items):
        self.user = pd.read_csv(features_dir / "user_features.csv").set_index("user_id")
        self.item = pd.read_csv(features_dir / "item_features.csv").set_index("item_id")
        self.genre_shares = pd.read_csv(features_dir / "user_genre_shares.csv", index_col=0)
        self.items = items

    def user_features(self, user_ids):
        return self.user.reindex(user_ids)

    def item_features(self, item_ids):
        return self.item.reindex(item_ids)

    def genre_affinity(self, user_ids, item_ids):
        return genre_affinity(user_ids, item_ids, self.genre_shares, self.items)

    def eligible_items(self, min_ratings):
        """Items with enough training ratings to be recommended."""
        return self.item.index[self.item["item_popularity"] >= min_ratings].to_numpy()
