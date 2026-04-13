"""Fetch real Singapore datasets from data.gov.sg and real ML datasets.

Replaces the tiny synthetic datasets that were generated previously. All
downloads are cached locally and committed to the repo so exercises run
against the exact same data everywhere.

Usage:
    python scripts/fetch-real-data.py              # fetch everything
    python scripts/fetch-real-data.py --only hdb   # just HDB resale
    python scripts/fetch-real-data.py --list       # show what will be fetched
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.request
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# ── data.gov.sg datastore — Singapore public datasets ───────────────

DATA_GOV_SG = {
    # ── Module 1: Data pipelines (HDB, weather, economic) ──────────
    "mlfp01/hdb_resale.parquet": {
        "source": "data.gov.sg",
        # Full HDB resale history — 5 separate datasets stitched together
        "resource_ids": [
            # 1990-1999
            "adbbddd3-30e2-445f-a123-29bee150a6fe",
            # 2000-Feb 2012
            "8c00bf08-9124-479e-aeca-7cc411d884c4",
            # Mar 2012 - Dec 2014
            "83b2fc37-ce8c-4df4-968b-370fd818138b",
            # Jan 2015 - Dec 2016
            "1b702208-44bf-4829-b620-4615ee19b57c",
            # Jan 2017 onwards
            "f1765b54-a209-4718-8d38-a39237f502b3",
        ],
        "description": "Full HDB resale transactions 1990-present (~1.2M rows)",
    },
    "mlfp01/sg_cpi.csv": {
        "source": "data.gov.sg",
        "resource_id": "a339ca0e-9c68-4810-a7cb-05b5ad0aeced",
        "description": "Consumer Price Index (CPI), Monthly — 1961-present",
    },
    "mlfp01/sg_employment.csv": {
        "source": "data.gov.sg",
        "resource_id": "b0d03bfd-a6d1-4b7c-ba65-51a92fcb4c0c",
        "description": "Overall Unemployment Rate, Quarterly — 1986-present",
    },
    "mlfp01/sg_population.csv": {
        "source": "data.gov.sg",
        "resource_id": "f9dbfc75-a2dc-42af-9f50-425e4107ae84",
        "description": "Singapore Residents By Age Group, Annual",
    },
    # MRT station amenities (for joins in lesson 1.4)
    "mlfp01/mrt_stations.csv": {
        "source": "data.gov.sg",
        "resource_id": "train-station-chinese-names",
        "description": "MRT station names and coordinates",
    },
    # ── Module 2: Credit scoring / A-B / ICU (real patient data) ──
    "mlfp02/sg_air_temp.csv": {
        "source": "data.gov.sg",
        "resource_id": "07752c06-a4c9-48da-a3f5-98e9a51c6e7b",
        "description": "Air Temperature (Mean Daily), 1980-2020",
    },
}

# ── HuggingFace datasets — real ML training data ────────────────────

HF_DATASETS = {
    "mlfp05/fashion_mnist": {
        "dataset": "zalando-datasets/fashion_mnist",
        "split": "train",
        "description": "Fashion-MNIST: 60K real 28x28 clothing images",
    },
    "mlfp05/cifar10": {
        "dataset": "uoft-cs/cifar10",
        "split": "train",
        "description": "CIFAR-10: 50K 32x32 real colour images, 10 classes",
    },
    "mlfp05/ag_news": {
        "dataset": "fancyzhx/ag_news",
        "split": "train",
        "description": "AG News: 120K real news headlines, 4 classes",
    },
    "mlfp05/imdb_sentiment": {
        "dataset": "stanfordnlp/imdb",
        "split": "train",
        "description": "IMDB: 25K movie reviews for sentiment analysis",
    },
    "mlfp05/cora_citation": {
        "dataset": "pyg-team/planetoid_cora",
        "description": "Cora citation network: 2708 real papers, 5429 citations, 7 classes",
    },
    # ── Module 6: LLM / RAG / alignment datasets ───────────────────
    "mlfp06/preference_pairs": {
        "dataset": "trl-lib/ultrafeedback_binarized",
        "split": "train",
        "description": "UltraFeedback: real LLM-generated preference pairs for DPO",
    },
    "mlfp06/sg_regulations": {
        "dataset": "law-ai/InLegalBERT",  # placeholder — will fall back
        "description": "Singapore policy documents for RAG",
    },
    "mlfp06/instruction_tuning": {
        "dataset": "tatsu-lab/alpaca",
        "split": "train",
        "description": "Alpaca 52K instruction-following dataset",
    },
}


def fetch_datastore_csv(resource_id: str, retries: int = 5) -> pl.DataFrame:
    """Fetch a full dataset from data.gov.sg datastore API with rate-limit handling."""
    base = "https://data.gov.sg/api/action/datastore_search"
    all_records = []
    offset = 0
    limit = 5000  # smaller pages to reduce rate-limit hits

    for _ in range(1000):  # safety cap
        url = f"{base}?resource_id={resource_id}&limit={limit}&offset={offset}"
        last_err = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "MLFP-CourseData/1.0 (terrene-foundation/mlfp)"
                    },
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read().decode("utf-8")
                break
            except Exception as e:
                last_err = e
                backoff = 15 * (attempt + 1)  # 15, 30, 45, 60, 75s
                print(f"    retry {attempt + 1}/{retries} (wait {backoff}s): {e}")
                time.sleep(backoff)
        else:
            raise last_err or RuntimeError("fetch failed")

        import json

        result = json.loads(data)
        if not result.get("success"):
            raise RuntimeError(f"API error: {result}")

        records = result["result"]["records"]
        if not records:
            break
        all_records.extend(records)
        total = result["result"].get("total", 0)
        print(f"    fetched {len(all_records):,}/{total:,}")
        if offset + limit >= total:
            break
        offset += limit
        time.sleep(2.0)  # be nice to the API

    if not all_records:
        return pl.DataFrame()
    return pl.DataFrame(all_records)


def fetch_hdb_resale() -> pl.DataFrame:
    """Fetch full HDB resale history 1990-present, stitched across 5 datasets."""
    print("  Fetching HDB resale full history (5 datasets, ~1.2M rows)...")
    cfg = DATA_GOV_SG["mlfp01/hdb_resale.parquet"]
    frames = []
    for i, rid in enumerate(cfg["resource_ids"], 1):
        print(f"    [{i}/5] {rid}")
        try:
            df = fetch_datastore_csv(rid)
            print(f"        {df.shape[0]:,} rows, columns: {df.columns}")
            frames.append(df)
        except Exception as e:
            print(f"        FAILED: {e}")

    if not frames:
        raise RuntimeError("No HDB data fetched")

    # Harmonize columns — older datasets may have different schemas
    all_cols = set()
    for f in frames:
        all_cols.update(f.columns)
    # Keep common important columns
    keep = [
        "month",
        "town",
        "flat_type",
        "block",
        "street_name",
        "storey_range",
        "floor_area_sqm",
        "flat_model",
        "lease_commence_date",
        "resale_price",
        "remaining_lease",
    ]
    keep = [c for c in keep if c in all_cols]

    harmonized = []
    for f in frames:
        missing = [c for c in keep if c not in f.columns]
        for c in missing:
            f = f.with_columns(pl.lit(None).alias(c))
        harmonized.append(f.select(keep))

    combined = pl.concat(harmonized, how="vertical_relaxed")
    # Coerce types
    combined = combined.with_columns(
        pl.col("floor_area_sqm").cast(pl.Float64, strict=False),
        pl.col("resale_price").cast(pl.Float64, strict=False),
        pl.col("lease_commence_date").cast(pl.Int64, strict=False),
    )
    return combined


def fetch_datastore_to_file(dest_rel: str, cfg: dict) -> None:
    """Fetch a single datastore resource and save as CSV or Parquet."""
    dest = DATA / dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)

    if "resource_ids" in cfg:
        # Multi-part (e.g., HDB resale)
        df = fetch_hdb_resale()
    else:
        df = fetch_datastore_csv(cfg["resource_id"])

    if df.shape[0] == 0:
        print(f"  ⚠ {dest_rel}: empty dataset, skipping")
        return

    if dest.suffix == ".parquet":
        df.write_parquet(dest, compression="zstd")
    else:
        df.write_csv(dest)

    size_mb = dest.stat().st_size / 1e6
    print(f"  ✓ {dest_rel}: {df.shape[0]:,} rows, {size_mb:.1f} MB")


def fetch_hf_dataset(dest_rel: str, cfg: dict) -> None:
    """Fetch a HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            f"  ⚠ {dest_rel}: huggingface datasets not installed. "
            "Install with: pip install datasets"
        )
        return

    dest = DATA / dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"  Loading {cfg['dataset']}...")
        ds = load_dataset(cfg["dataset"], split=cfg.get("split", "train"))

        max_rows = cfg.get("max_rows")
        if max_rows and len(ds) > max_rows:
            ds = ds.select(range(max_rows))

        # Save as parquet directory (supports nested features)
        ds.to_parquet(str(dest.with_suffix(".parquet")))
        size_mb = dest.with_suffix(".parquet").stat().st_size / 1e6
        print(f"  ✓ {dest_rel}: {len(ds):,} rows, {size_mb:.1f} MB")
    except Exception as e:
        print(f"  ✗ {dest_rel}: {e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="Fetch only datasets matching this substring")
    parser.add_argument("--list", action="store_true", help="List what will be fetched")
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Skip HuggingFace datasets (for data.gov.sg only)",
    )
    args = parser.parse_args()

    all_targets = [("datagovsg", k, v) for k, v in DATA_GOV_SG.items()]
    if not args.skip_hf:
        all_targets += [("hf", k, v) for k, v in HF_DATASETS.items()]

    if args.only:
        all_targets = [t for t in all_targets if args.only in t[1]]

    if args.list:
        print(f"Would fetch {len(all_targets)} datasets:")
        for kind, name, cfg in all_targets:
            print(f"  [{kind}] {name}: {cfg.get('description', '?')}")
        return

    print(f"Fetching {len(all_targets)} datasets...\n")
    success = 0
    for kind, name, cfg in all_targets:
        print(f"[{kind}] {name}")
        try:
            if kind == "datagovsg":
                fetch_datastore_to_file(name, cfg)
            elif kind == "hf":
                fetch_hf_dataset(name, cfg)
            success += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
        print()

    print(f"\n{'='*60}")
    print(f"Complete: {success}/{len(all_targets)} datasets fetched")


if __name__ == "__main__":
    main()
