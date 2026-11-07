"""Deliberately broken records used to show that each validation check works.

MovieLens 100K is a clean, curated dataset, so most checks would find nothing.
When `inject_test_errors` is true, these rows are appended during ingestion with
record_source = "injected_test". Validation does not look at that column; it is
only used afterwards to report how many of the injected problems were caught.
"""

import pandas as pd

INJECTED = "injected_test"


def _rows(template, changes):
    rows = []
    for change in changes:
        row = template.copy()
        row.update(change)
        rows.append(row)
    return pd.DataFrame(rows)


def inject_test_errors(users, items, interactions):
    first_user = users.iloc[0].to_dict()
    first_item = items.iloc[0].to_dict()
    first_event = interactions.iloc[0].to_dict()

    bad_users = _rows(first_user, [
        {"age": "33"},                                   # duplicate user_id
        {"user_id": "5001", "gender": None},             # missing value
        {"user_id": "5002", "age": "250"},               # age out of range
        {"user_id": "5003", "gender": "X"},              # invalid category
        {"user_id": "5004", "age": "thirty"},            # wrong data type
    ])
    bad_items = _rows(first_item, [
        {"title": "Duplicate entry"},                    # duplicate item_id
        {"item_id": "9001", "title": None},              # missing title
    ])
    bad_events = _rows(first_event, [
        {},                                              # exact duplicate row
        {"rating": "0"},                                 # rating below range
        {"rating": "7"},                                 # rating above range
        {"rating": "4.5"},                               # rating not a whole star
        {"rating": "five"},                              # wrong data type
        {"rating": None},                                # missing rating
        {"timestamp": "not_a_time"},                     # unparseable timestamp
        {"timestamp": "2051222400"},                     # 2035, in the future
        {"timestamp": "100000000"},                      # 1973, before the platform existed
        {"user_id": "99999"},                            # unknown user (referential integrity)
        {"item_id": "99999"},                            # unknown item (referential integrity)
        {"user_id": "abc"},                              # wrong data type
    ])

    def tag(df, source):
        df = df.copy()
        df["record_source"] = source
        return df

    users = pd.concat([tag(users, "dataset"), tag(bad_users, INJECTED)], ignore_index=True)
    items = pd.concat([tag(items, "dataset"), tag(bad_items, INJECTED)], ignore_index=True)
    interactions = pd.concat([tag(interactions, "dataset"), tag(bad_events, INJECTED)], ignore_index=True)
    return users, items, interactions
