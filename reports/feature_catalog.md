# Feature catalog

Generated from `src/features/registry.py`. Training features are computed from training interactions only, with the reference time set to the train/test cutoff.

| feature | entity | type | description | source | available at prediction | missing % |
| --- | --- | --- | --- | --- | --- | --- |
| user_activity_count | user | int | Number of ratings given by the user | interactions | yes | 0.00 |
| user_avg_rating | user | float | Mean rating given by the user | interactions | yes | 0.00 |
| user_rating_std | user | float | Standard deviation of the user's ratings (0 if one rating) | interactions | yes | 0.00 |
| user_days_since_last_interaction | user | int | Days between the user's last rating and the feature reference time | interactions | yes | 0.00 |
| user_preferred_genre | user | category | Genre the user rated most often (ties: alphabetical) | interactions + items | yes | 0.00 |
| user_genre_count | user | int | Number of distinct genres the user has rated | interactions + items | yes | 0.00 |
| user_age_group | user | category | Age band: <18, 18-24, 25-34, 35-44, 45-54, 55+ | users | yes | 0.00 |
| user_gender | user | category | Gender as given in the user profile (M/F) | users | yes | 0.00 |
| user_occupation | user | category | Occupation from the user profile | users | yes | 0.00 |
| item_popularity | item | int | Number of ratings the item received | interactions | yes | 0.00 |
| item_avg_rating | item | float | Mean rating of the item | interactions | yes | 3.92 |
| item_bayes_rating | item | float | Average rating shrunk towards the global mean: (n*avg + m*global)/(n + m) | interactions | yes | 0.00 |
| item_days_since_last_interaction | item | int | Days between the item's last rating and the feature reference time | interactions | yes | 3.92 |
| item_release_year | item | int | Release year (median year if missing) | items | yes | 0.00 |
| item_primary_genre | item | category | First genre listed for the item | items | yes | 0.00 |
| item_genre_count | item | int | Number of genres attached to the item | items | yes | 0.00 |
| genre_affinity | user-item | float | Average share of the user's past ratings that fall in the item's genres | interactions + items | yes | 0.00 |
| rating_deviation | user-item | float | Rating minus the user's average rating (positive = liked more than usual) | interactions | training only | 0.00 |
| recency_weight | user-item | float | exp(-days_before_reference / 30); recent interactions weigh more | interactions | training only | 0.00 |
