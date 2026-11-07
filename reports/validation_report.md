# Data validation report

Generated: 2026-11-07T09:52:21.658404+00:00  
Source: `movielens_100k`

Rows that fail an **error** check are removed and saved to `data/errors/<table>_errors.csv` with the failed checks listed. **Warnings** are kept and handled in data preparation.

## Overview

| table | input rows | valid rows | rejected rows |
| --- | --- | --- | --- |
| users | 948 | 943 | 5 |
| items | 1684 | 1682 | 2 |
| interactions | 100012 | 100000 | 12 |

## Users

| check | severity | description | failed_rows |
| --- | --- | --- | --- |
| missing_value | error | user_id, age, gender or occupation is empty | 1 |
| invalid_type | error | user_id or age is not an integer | 1 |
| duplicate_id | error | user_id appears more than once (first kept) | 1 |
| age_out_of_range | error | age outside 1-100 | 1 |
| invalid_gender | error | gender not in ['M', 'F'] | 1 |

Injected test records: 5, caught: 5. Rows from the dataset itself rejected: 0.

## Items

| check | severity | description | failed_rows |
| --- | --- | --- | --- |
| missing_value | error | item_id or title is empty | 1 |
| invalid_type | error | item_id is not an integer | 0 |
| duplicate_id | error | item_id appears more than once (first kept) | 1 |
| missing_release_date | warning | release date is empty (kept, imputed later) | 1 |
| missing_genre | warning | no genre assigned (kept, labelled 'unknown') | 0 |

Injected test records: 2, caught: 2. Rows from the dataset itself rejected: 0.

## Interactions

| check | severity | description | failed_rows |
| --- | --- | --- | --- |
| missing_value | error | user_id, item_id, rating or timestamp is empty | 1 |
| invalid_type | error | user_id or item_id is not an integer, or rating is not numeric | 2 |
| duplicate_record | error | exact duplicate of an earlier interaction (first kept) | 1 |
| invalid_rating | error | rating not a whole number in 1-5 | 3 |
| invalid_timestamp | error | timestamp unparseable, before 1995-01-01 or in the future | 3 |
| unknown_user | error | user_id not found in the valid users table | 1 |
| unknown_item | error | item_id not found in the valid items table | 1 |

Injected test records: 12, caught: 12. Rows from the dataset itself rejected: 0.
