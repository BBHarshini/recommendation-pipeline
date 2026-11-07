"""Stage 2: Data validation.

Runs automated checks on the raw tables. Valid rows go to data/validated, invalid
rows go to data/errors with the reason(s) they failed, and a report is written to
reports/validation_report.md (+ .json).
"""

from datetime import datetime, timezone

import pandas as pd

from src.common import get_logger, markdown_table, project_path, write_json
from src.validation.checks import apply_checks, interaction_checks, item_checks, user_checks


def _injection_summary(raw, errors):
    """How many deliberately broken rows were caught, and how many real rows were rejected."""
    if "record_source" not in raw.columns:
        return None
    injected = int((raw["record_source"] == "injected_test").sum())
    caught = int((errors["record_source"] == "injected_test").sum()) if len(errors) else 0
    real_rejected = int((errors["record_source"] != "injected_test").sum()) if len(errors) else 0
    return {"injected": injected, "caught": caught, "dataset_rows_rejected": real_rejected}


def run(cfg):
    logger = get_logger(cfg)
    v = cfg["validation"]
    raw_dir = project_path(cfg["data"]["raw_dir"])
    out_dir = project_path(cfg["data"]["validated_dir"])
    err_dir = project_path(cfg["data"]["errors_dir"])
    reports_dir = project_path(cfg["paths"]["reports_dir"])

    raw = {name: pd.read_csv(raw_dir / f"{name}.csv", dtype=str, keep_default_na=True)
           for name in ["users", "items", "interactions"]}

    report = {"validated_at": datetime.now(timezone.utc).isoformat(), "tables": {}}

    # users and items first: interactions are checked against the *valid* ids
    clean_users, user_err, user_sum = apply_checks(raw["users"], user_checks(v))
    clean_items, item_err, item_sum = apply_checks(raw["items"], item_checks(v))
    valid_users = pd.to_numeric(clean_users["user_id"]).astype(int)
    valid_items = pd.to_numeric(clean_items["item_id"]).astype(int)
    clean_events, event_err, event_sum = apply_checks(
        raw["interactions"], interaction_checks(v, set(valid_users), set(valid_items)))

    for name, clean, err, summary in [("users", clean_users, user_err, user_sum),
                                      ("items", clean_items, item_err, item_sum),
                                      ("interactions", clean_events, event_err, event_sum)]:
        clean.to_csv(out_dir / f"{name}.csv", index=False)
        err.to_csv(err_dir / f"{name}_errors.csv", index=False)
        report["tables"][name] = {"input_rows": len(raw[name]), "valid_rows": len(clean),
                                  "rejected_rows": len(err), "checks": summary,
                                  "injected_test_records": _injection_summary(raw[name], err)}
        logger.info("Validation %-12s: %d rows in, %d valid, %d rejected",
                    name, len(raw[name]), len(clean), len(err))

    write_json(report, reports_dir / "validation_report.json")
    (reports_dir / "validation_report.md").write_text(_to_markdown(report, cfg), encoding="utf-8")
    return {name: t["rejected_rows"] for name, t in report["tables"].items()}


def _to_markdown(report, cfg):
    lines = ["# Data validation report", "",
             f"Generated: {report['validated_at']}  ",
             f"Source: `{cfg['data']['source']}`", "",
             "Rows that fail an **error** check are removed and saved to `data/errors/<table>_errors.csv` "
             "with the failed checks listed. **Warnings** are kept and handled in data preparation.", ""]
    overview = pd.DataFrame([{"table": n, "input rows": t["input_rows"], "valid rows": t["valid_rows"],
                              "rejected rows": t["rejected_rows"]} for n, t in report["tables"].items()])
    lines += ["## Overview", "", markdown_table(overview), ""]

    for name, t in report["tables"].items():
        lines += [f"## {name.capitalize()}", "", markdown_table(pd.DataFrame(t["checks"])), ""]
        inj = t["injected_test_records"]
        if inj:
            lines += [f"Injected test records: {inj['injected']}, caught: {inj['caught']}. "
                      f"Rows from the dataset itself rejected: {inj['dataset_rows_rejected']}.", ""]
    return "\n".join(lines)
