"""Row-level data quality checks.

Every check returns a boolean Series (True = the row fails). Checks marked as
errors remove the row from the clean data; warnings keep the row but are reported.
"""

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass
class Check:
    name: str
    description: str
    severity: str          # "error" or "warning"
    func: Callable


def is_missing(df, columns):
    return df[columns].isna().any(axis=1) | df[columns].astype(str).apply(lambda c: c.str.strip() == "").any(axis=1)


def not_integer(series):
    """Present but not a whole number (e.g. 'abc', '4.5')."""
    num = pd.to_numeric(series, errors="coerce")
    present = series.notna() & (series.astype(str).str.strip() != "")
    return present & (num.isna() | (num % 1 != 0))


def as_number(series):
    return pd.to_numeric(series, errors="coerce")


def user_checks(v):
    return [
        Check("missing_value", "user_id, age, gender or occupation is empty", "error",
              lambda d: is_missing(d, ["user_id", "age", "gender", "occupation"])),
        Check("invalid_type", "user_id or age is not an integer", "error",
              lambda d: not_integer(d["user_id"]) | not_integer(d["age"])),
        Check("duplicate_id", "user_id appears more than once (first kept)", "error",
              lambda d: d["user_id"].duplicated(keep="first")),
        Check("age_out_of_range", f"age outside {v['age_min']}-{v['age_max']}", "error",
              lambda d: as_number(d["age"]).notna() & ~as_number(d["age"]).between(v["age_min"], v["age_max"])),
        Check("invalid_gender", f"gender not in {v['allowed_genders']}", "error",
              lambda d: d["gender"].notna() & ~d["gender"].isin(v["allowed_genders"])),
    ]


def item_checks(v):
    return [
        Check("missing_value", "item_id or title is empty", "error",
              lambda d: is_missing(d, ["item_id", "title"])),
        Check("invalid_type", "item_id is not an integer", "error",
              lambda d: not_integer(d["item_id"])),
        Check("duplicate_id", "item_id appears more than once (first kept)", "error",
              lambda d: d["item_id"].duplicated(keep="first")),
        Check("missing_release_date", "release date is empty (kept, imputed later)", "warning",
              lambda d: is_missing(d, ["release_date"])),
        Check("missing_genre", "no genre assigned (kept, labelled 'unknown')", "warning",
              lambda d: is_missing(d, ["genres"])),
    ]


def interaction_checks(v, valid_user_ids, valid_item_ids):
    earliest = pd.Timestamp(v["earliest_timestamp"], tz="UTC").timestamp()
    latest = pd.Timestamp.now(tz="UTC").timestamp()
    key = ["user_id", "item_id", "rating", "timestamp"]

    def bad_rating(d):
        r = as_number(d["rating"])
        return r.notna() & ((r % 1 != 0) | ~r.between(v["rating_min"], v["rating_max"]))

    def bad_timestamp(d):
        ts = as_number(d["timestamp"])
        present = d["timestamp"].notna()
        return present & (ts.isna() | (ts < earliest) | (ts > latest))

    return [
        Check("missing_value", "user_id, item_id, rating or timestamp is empty", "error",
              lambda d: is_missing(d, key)),
        Check("invalid_type", "user_id or item_id is not an integer, or rating is not numeric", "error",
              lambda d: not_integer(d["user_id"]) | not_integer(d["item_id"])
              | (d["rating"].notna() & as_number(d["rating"]).isna())),
        Check("duplicate_record", "exact duplicate of an earlier interaction (first kept)", "error",
              lambda d: d.duplicated(subset=key, keep="first")),
        Check("invalid_rating", f"rating not a whole number in {v['rating_min']}-{v['rating_max']}", "error",
              bad_rating),
        Check("invalid_timestamp", f"timestamp unparseable, before {v['earliest_timestamp']} or in the future",
              "error", bad_timestamp),
        Check("unknown_user", "user_id not found in the valid users table", "error",
              lambda d: as_number(d["user_id"]).notna() & ~as_number(d["user_id"]).isin(valid_user_ids)),
        Check("unknown_item", "item_id not found in the valid items table", "error",
              lambda d: as_number(d["item_id"]).notna() & ~as_number(d["item_id"]).isin(valid_item_ids)),
    ]


def apply_checks(df, checks):
    """Run all checks; return (clean rows, error rows with reasons, per-check summary)."""
    results = {c.name: c.func(df).fillna(False).astype(bool) for c in checks}
    summary = [{"check": c.name, "severity": c.severity, "description": c.description,
                "failed_rows": int(results[c.name].sum())} for c in checks]

    error_names = [c.name for c in checks if c.severity == "error"]
    failed = pd.DataFrame({n: results[n] for n in error_names})
    is_error = failed.any(axis=1)

    errors = df[is_error].copy()
    errors["failed_checks"] = failed[is_error].apply(lambda r: ",".join(r.index[r]), axis=1)
    return df[~is_error].copy(), errors, summary
