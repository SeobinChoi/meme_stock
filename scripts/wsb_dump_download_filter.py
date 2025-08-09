#!/usr/bin/env python3
"""
Download monthly Reddit submissions dumps for 2022-2023, filter to r/wallstreetbets,
write original rows to CSV per-year, and build a daily aggregated CSV for loader use.

Outputs:
- data/raw/wsb_submissions_2022.csv
- data/raw/wsb_submissions_2023.csv
- data/processed/wsb_daily_aggregated_2022_2023.csv

Notes:
- Uses Internet Archive / Pushshift mirrors. Falls back automatically.
- Does not run sentiment or keyword extraction; only raw subset + daily aggregates.
"""

from __future__ import annotations

import os
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import requests
import zstandard as zstd


MONTHS: List[str] = [f"{m:02d}" for m in range(1, 13)]
YEARS: List[str] = ["2022", "2023"]

# 1st priority: Internet Archive; 2nd: pushshift mirror (availability varies)
URLS: List[str] = [
    "https://archive.org/download/pushshift_rs/RS_{Y}-{M}.zst",
    "https://files.pushshift.io/reddit/submissions/RS_{Y}-{M}.zst",
]

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
AGG_OUT = PROC_DIR / "wsb_daily_aggregated_2022_2023.csv"

SUBREDDIT_SET = {"wallstreetbets", "WallStreetBets", "WALLSTREETBETS"}


def stream_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 512):
                if chunk:
                    f.write(chunk)


def iter_zst_jsonlines(zst_path: Path):
    dctx = zstd.ZstdDecompressor()
    with open(zst_path, "rb") as fh, dctx.stream_reader(fh) as reader:
        buf = b""
        while True:
            chunk = reader.read(1024 * 1024)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def monthly_process(year: str, month: str) -> Optional[Path]:
    zst_tmp = RAW_DIR / f"RS_{year}-{month}.zst"
    out_csv = RAW_DIR / f"wsb_submissions_{year}_{month}.csv"

    for base in URLS:
        url = base.format(Y=year, M=month)
        try:
            print(f"↓ downloading {url}")
            stream_download(url, zst_tmp)
            break
        except Exception as e:
            print(f"  fallback due to: {e}")
            continue
    else:
        print(f"× failed to download month {year}-{month}")
        return None

    kept = 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Match raw_reddit_wsb.csv header style for compatibility
        w.writerow(["title", "score", "id", "url", "comms_num", "created", "body", "timestamp"])
        for obj in iter_zst_jsonlines(zst_tmp):
            sub = obj.get("subreddit")
            if sub not in SUBREDDIT_SET:
                continue
            title = obj.get("title", "")
            score = obj.get("score", 0)
            sid = obj.get("id", "")
            url = obj.get("url") or obj.get("full_link") or ""
            comms = obj.get("num_comments", 0)
            created = obj.get("created_utc", None)
            body = obj.get("selftext", "")
            ts_str = ""
            if created is not None:
                try:
                    ts_str = datetime.utcfromtimestamp(int(created)).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    ts_str = ""
            w.writerow([title, score, sid, url, comms, created, body, ts_str])
            kept += 1

    print(f"✓ saved {out_csv} (rows={kept})")
    try:
        zst_tmp.unlink()
    except Exception:
        pass
    return out_csv


def build_year_csv(year: str) -> Optional[Path]:
    parts: List[Path] = []
    for m in MONTHS:
        p = monthly_process(year, m)
        if p:
            parts.append(p)
    if not parts:
        return None

    year_csv = RAW_DIR / f"wsb_submissions_{year}.csv"
    with open(year_csv, "w", newline="", encoding="utf-8") as wfh:
        w = csv.writer(wfh)
        wrote_header = False
        for part in parts:
            with open(part, "r", encoding="utf-8") as rfh:
                reader = csv.reader(rfh)
                header = next(reader, None)
                if header and not wrote_header:
                    w.writerow(header)
                    wrote_header = True
                for row in reader:
                    w.writerow(row)
            try:
                os.remove(part)
            except Exception:
                pass
    print(f"✓ saved {year_csv}")
    return year_csv


def aggregate_daily(year_csvs: List[Optional[Path]]) -> None:
    import pandas as pd

    dfs = []
    for ycsv in year_csvs:
        if ycsv is None:
            continue
        df = pd.read_csv(ycsv)
        df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
        df["comms_num"] = pd.to_numeric(df["comms_num"], errors="coerce").fillna(0).astype(int)
        g = (
            df.groupby("date")
            .agg(
                score_sum=("score", "sum"),
                score_count=("score", "count"),
                score_mean=("score", "mean"),
                num_comments_sum=("comms_num", "sum"),
                num_comments_mean=("comms_num", "mean"),
            )
            .reset_index()
        )
        dfs.append(g)

    if not dfs:
        print("× no yearly CSVs to aggregate")
        return

    full = __import__("pandas").concat(dfs, ignore_index=True)
    full = full.groupby("date", as_index=False).sum()
    full["date"] = __import__("pandas").to_datetime(full["date"]).dt.strftime("%Y-%m-%d")
    AGG_OUT.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(AGG_OUT, index=False)
    print(f"✓ saved aggregated daily: {AGG_OUT} (rows={len(full)})")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    years_out: List[Optional[Path]] = []
    for y in YEARS:
        years_out.append(build_year_csv(y))
    aggregate_daily(years_out)


if __name__ == "__main__":
    main()


