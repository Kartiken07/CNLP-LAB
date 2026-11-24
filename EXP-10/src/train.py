"""Fine-tune NLLB-200 (distilled 600M) on English-Assamese sentence pairs.

    python src/train.py --data-dir data/processed --output-dir nllb-en-as
Reduce --batch-size or --max-length if the GPU runs out of memory.
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)

LANG_CODES = {"en": "eng_Latn", "as": "asm_Beng"}


def build_dataset(path, tokenizer, max_length, max_rows=None):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if max_rows:
        df = df.head(max_rows)
    ds = Dataset.from_pandas(df[["src", "tgt"]])

    def tok(batch):
        return tokenizer(batch["src"], text_target=batch["tgt"], max_length=max_length, truncation=True)

    return ds.map(tok, batched=True, remove_columns=["src", "tgt"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--output-dir", default="nllb-en-as")
    ap.add_argument("--direction", choices=["en-as", "as-en"], default="en-as")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train-rows", type=int, help="use a small subset for smoke tests")
    ap.add_argument("--max-val-rows", type=int)
    args = ap.parse_args()

    set_seed(args.seed)
    print("GPU available:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    s, t = args.direction.split("-")
    tokenizer = AutoTokenizer.from_pretrained(args.model, src_lang=LANG_CODES[s], tgt_lang=LANG_CODES[t])
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    # for as-en the processed CSV columns are swapped
    def load(name, rows):
        ds = build_dataset(f"{args.data_dir}/{name}.csv", tokenizer, args.max_length, rows) if s == "en" else None
        if ds is None:
            df = pd.read_csv(f"{args.data_dir}/{name}.csv", dtype=str, keep_default_na=False).rename(columns={"src": "tgt", "tgt": "src"})
            if rows:
                df = df.head(rows)
            tmp = f"/tmp/{name}_swapped.csv"; df.to_csv(tmp, index=False)
            ds = build_dataset(tmp, tokenizer, args.max_length)
        return ds

    train_ds, val_ds = load("train", args.max_train_rows), load("validation", args.max_val_rows)
    print(f"train={len(train_ds):,} validation={len(val_ds):,}")

    import dataclasses
    # newer transformers versions replaced warmup_ratio by a float warmup_steps (< 1 means a ratio)
    fields = {f.name for f in dataclasses.fields(Seq2SeqTrainingArguments)}
    warmup = {"warmup_ratio": 0.03} if "warmup_ratio" in fields else {"warmup_steps": 0.03}
    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, weight_decay=0.01,
        **warmup, eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=50, fp16=torch.cuda.is_available(), seed=args.seed, report_to="none",
    )
    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                             data_collator=DataCollatorForSeq2Seq(tokenizer, model=model), processing_class=tokenizer)
    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60
    print(f"Training time: {minutes:.1f} min")
    trainer.save_model(args.output_dir); tokenizer.save_pretrained(args.output_dir)
    pd.DataFrame(trainer.state.log_history).to_csv(f"{args.output_dir}/training_log.csv", index=False)
    with open(f"{args.output_dir}/training_time_minutes.txt", "w") as f:
        f.write(f"{minutes:.1f}\n")


if __name__ == "__main__":
    main()
