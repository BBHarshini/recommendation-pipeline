"""Feature catalogue: one definition per feature.

This list is the single source of truth. The feature builder computes exactly these
columns, and reports/feature_catalog.md is generated from it, so the documentation
cannot drift away from the code.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    name: str
    entity: str            # user, item or user-item
    dtype: str
    description: str
    source: str
    at_prediction: bool    # can it be computed when making a live recommendation?


FEATURES = [
    # ---- user features
    Feature("user_activity_count", "user", "int", "Number of ratings given by the user", "interactions", True),
    Feature("user_avg_rating", "user", "float", "Mean rating given by the user", "interactions", True),
    Feature("user_rating_std", "user", "float", "Standard deviation of the user's ratings (0 if one rating)",
            "interactions", True),
    Feature("user_days_since_last_interaction", "user", "int",
            "Days between the user's last rating and the feature reference time", "interactions", True),
    Feature("user_preferred_genre", "user", "category",
            "Genre the user rated most often (ties: alphabetical)", "interactions + items", True),
    Feature("user_genre_count", "user", "int", "Number of distinct genres the user has rated",
            "interactions + items", True),
    Feature("user_age_group", "user", "category", "Age band: <18, 18-24, 25-34, 35-44, 45-54, 55+", "users", True),
    Feature("user_gender", "user", "category", "Gender as given in the user profile (M/F)", "users", True),
    Feature("user_occupation", "user", "category", "Occupation from the user profile", "users", True),
    # ---- item features
    Feature("item_popularity", "item", "int", "Number of ratings the item received", "interactions", True),
    Feature("item_avg_rating", "item", "float", "Mean rating of the item", "interactions", True),
    Feature("item_bayes_rating", "item", "float",
            "Average rating shrunk towards the global mean: (n*avg + m*global)/(n + m)", "interactions", True),
    Feature("item_days_since_last_interaction", "item", "int",
            "Days between the item's last rating and the feature reference time", "interactions", True),
    Feature("item_release_year", "item", "int", "Release year (median year if missing)", "items", True),
    Feature("item_primary_genre", "item", "category", "First genre listed for the item", "items", True),
    Feature("item_genre_count", "item", "int", "Number of genres attached to the item", "items", True),
    # ---- user-item (interaction) features
    Feature("genre_affinity", "user-item", "float",
            "Average share of the user's past ratings that fall in the item's genres", "interactions + items", True),
    Feature("rating_deviation", "user-item", "float",
            "Rating minus the user's average rating (positive = liked more than usual)", "interactions", False),
    Feature("recency_weight", "user-item", "float",
            "exp(-days_before_reference / 30); recent interactions weigh more", "interactions", False),
]


def names(entity):
    return [f.name for f in FEATURES if f.entity == entity]
