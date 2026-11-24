"""Load, clean, deduplicate and split English-Assamese parallel data.

Examples
--------
Open dataset downloaded automatically from Hugging Face (recommended, no manual files):
    python src/preprocess.py --hf-dataset samanantar --max-rows 40000 --out-dir data/processed

Two aligned plain-text files (one sentence per line):
    python src/preprocess.py --src-file train.en --tgt-file train.as --out-dir data/processed

A single TSV/CSV/JSONL file with an English and an Assamese column:
    python src/preprocess.py --input train.tsv --src-col en --tgt-col as --out-dir data/processed

Exclude any pair that also appears in test/reference data (prevents leakage):
    python src/preprocess.py ... --exclude wmt_test.tsv --exclude-src-col en
"""
import argparse
import os
import re
import unicodedata

import pandas as pd

SEED = 42
_CONTROL = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F​‎‏‪-‮﻿]")
_SPACES = re.compile(r"\s+")


def clean_text(text) -> str:
    """NFC-normalise, drop control/invisible characters, collapse whitespace.

    Assamese script and punctuation are preserved. ZWJ/ZWNJ (U+200C/U+200D) are
    kept because they can be meaningful in Indic scripts.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def read_lines(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def load_table(path: str, src_col: str, tgt_col: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tsv", ".txt"):
        df = pd.read_csv(path, sep="\t", dtype=str, quoting=3, keep_default_na=False)
    elif ext == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif ext in (".jsonl", ".json"):
        df = pd.read_json(path, lines=ext == ".jsonl", dtype=False)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    missing = [c for c in (src_col, tgt_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Columns {missing} not found. Available: {list(df.columns)}")
    return df[[src_col, tgt_col]].rename(columns={src_col: "src", tgt_col: "tgt"})


HF_DATASETS = {
    # AI4Bharat Samanantar, config "as": fields src (English) / tgt (Assamese), ~141k pairs, CC BY-NC 4.0
    "samanantar": ("ai4bharat/samanantar", "as"),
    # OPUS-100, config "as-en": translation dict with keys "as"/"en", ~142k pairs, licence unspecified
    "opus100": ("Helsinki-NLP/opus-100", "as-en"),
}


def records_to_df(records, kind: str) -> pd.DataFrame:
    """Convert an iterable of Hugging Face rows into a src(English)/tgt(Assamese) DataFrame."""
    if kind == "samanantar":
        return pd.DataFrame({"src": [r["src"] for r in records], "tgt": [r["tgt"] for r in records]})
    if kind == "opus100":
        return pd.DataFrame({"src": [r["translation"]["en"] for r in records],
                             "tgt": [r["translation"]["as"] for r in records]})
    raise ValueError(f"Unknown dataset: {kind}")


def load_hf(kind: str) -> pd.DataFrame:
    from datasets import load_dataset  # imported lazily so the rest works without it

    name, config = HF_DATASETS[kind]
    ds = load_dataset(name, config)
    frames = [records_to_df(ds[split], kind) for split in ds]  # use every split; we re-split ourselves
    return pd.concat(frames, ignore_index=True)


def load_parallel(args) -> pd.DataFrame:
    if args.hf_dataset:
        return load_hf(args.hf_dataset)
    if args.src_file and args.tgt_file:
        src, tgt = read_lines(args.src_file), read_lines(args.tgt_file)
        if len(src) != len(tgt):
            raise ValueError(f"Line counts differ: {len(src)} source vs {len(tgt)} target lines. Files are not aligned.")
        return pd.DataFrame({"src": src, "tgt": tgt})
    if args.input:
        return load_table(args.input, args.src_col, args.tgt_col)
    raise SystemExit("Provide --hf-dataset, or --src-file and --tgt-file, or --input.")


def clean_pairs(df: pd.DataFrame, min_chars: int = 1, max_chars: int = 1000, max_len_ratio: float = 4.0):
    """Return (clean_df, stats). Removes empty, missing, duplicate and badly misaligned pairs."""
    stats = {"raw": len(df)}
    df = df.copy()
    df["src"] = df["src"].map(clean_text)
    df["tgt"] = df["tgt"].map(clean_text)
    df = df[(df.src.str.len() >= min_chars) & (df.tgt.str.len() >= min_chars)]
    stats["after_empty_removal"] = len(df)
    df = df[(df.src.str.len() <= max_chars) & (df.tgt.str.len() <= max_chars)]
    stats["after_length_filter"] = len(df)
    ratio = (df.src.str.len() / df.tgt.str.len()).clip(lower=1e-9)
    df = df[(ratio <= max_len_ratio) & (ratio >= 1 / max_len_ratio)]
    stats["after_ratio_filter"] = len(df)
    df = df.drop_duplicates(subset=["src", "tgt"]).reset_index(drop=True)
    stats["after_dedup"] = len(df)
    return df, stats


def split_pairs(df: pd.DataFrame, seed: int = SEED, train: float = 0.8, val: float = 0.1):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train, n_val = int(n * train), int(n * val)
    return df[:n_train], df[n_train:n_train + n_val], df[n_train + n_val:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-dataset", choices=sorted(HF_DATASETS), help="download an open dataset from Hugging Face")
    ap.add_argument("--max-rows", type=int, help="keep a random subset of this many cleaned pairs (faster training)")
    ap.add_argument("--src-file"); ap.add_argument("--tgt-file")
    ap.add_argument("--input"); ap.add_argument("--src-col", default="en"); ap.add_argument("--tgt-col", default="as")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-chars", type=int, default=1000)
    ap.add_argument("--max-len-ratio", type=float, default=4.0, help="drop pairs whose char-length ratio exceeds this (misalignment guard)")
    ap.add_argument("--exclude", help="test/reference file whose sentence pairs must NOT be in the data")
    ap.add_argument("--exclude-src-col", default="en"); ap.add_argument("--exclude-tgt-col", default="as")
    args = ap.parse_args()

    df = load_parallel(args)
    df, stats = clean_pairs(df, max_chars=args.max_chars, max_len_ratio=args.max_len_ratio)
    if args.exclude:
        ex = load_table(args.exclude, args.exclude_src_col, args.exclude_tgt_col)
        ex_src = set(ex.src.map(clean_text)); ex_tgt = set(ex.tgt.map(clean_text))
        df = df[~df.src.isin(ex_src) & ~df.tgt.isin(ex_tgt)].reset_index(drop=True)
        stats["after_excluding_test_overlap"] = len(df)

    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
        stats["after_subsampling"] = len(df)

    tr, va, te = split_pairs(df, seed=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, part in (("train", tr), ("validation", va), ("test", te)):
        part.to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)
    stats.update(train=len(tr), validation=len(va), test=len(te))
    pd.Series(stats).to_csv(os.path.join(args.out_dir, "preprocess_stats.csv"), header=["count"])
    for k, v in stats.items():
        print(f"{k:32s} {v:>10,}")
    print("\nSample pairs (check alignment):")
    print(tr.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
