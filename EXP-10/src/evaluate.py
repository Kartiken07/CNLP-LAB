"""Evaluate a translation model on the held-out test split with BLEU and chrF.

Run once for the pretrained model and once for the fine-tuned one, on the SAME examples:
    python src/evaluate.py --model facebook/nllb-200-distilled-600M --label "Pretrained (before fine-tuning)"
    python src/evaluate.py --model nllb-en-as --label "Fine-tuned"
Rows are appended to results/metrics.csv; sample translations go to results/sample_translations.csv.
"""
import argparse
import os

import pandas as pd
import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

LANG_CODES = {"en": "eng_Latn", "as": "asm_Beng"}


def translate(model, tokenizer, sentences, tgt_code, batch_size, max_length, num_beams, device):
    out = []
    forced = tokenizer.convert_tokens_to_ids(tgt_code)
    for i in range(0, len(sentences), batch_size):
        batch = tokenizer(sentences[i:i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            gen = model.generate(**batch, forced_bos_token_id=forced, max_new_tokens=max_length, num_beams=num_beams)
        out += tokenizer.batch_decode(gen, skip_special_tokens=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--direction", choices=["en-as", "as-en"], default="en-as")
    ap.add_argument("--max-samples", type=int, help="evaluate on the first N test pairs (same N for both models!)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--n-samples", type=int, default=5)
    args = ap.parse_args()

    s, t = args.direction.split("-")
    df = pd.read_csv(f"{args.data_dir}/test.csv", dtype=str, keep_default_na=False)
    if s == "as":
        df = df.rename(columns={"src": "tgt", "tgt": "src"})
    if args.max_samples:
        df = df.head(args.max_samples)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, src_lang=LANG_CODES[s], tgt_lang=LANG_CODES[t])
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()

    hyps = translate(model, tokenizer, df.src.tolist(), LANG_CODES[t], args.batch_size, args.max_length, args.num_beams, device)
    refs = df.tgt.tolist()
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    print(f"{args.label}: BLEU={bleu.score:.2f} chrF={chrf.score:.2f} on {len(df)} test pairs")
    print(bleu)

    os.makedirs(args.results_dir, exist_ok=True)
    row = pd.DataFrame([{"model": args.label, "direction": args.direction, "test_pairs": len(df),
                         "BLEU": round(bleu.score, 2), "chrF": round(chrf.score, 2)}])
    mpath = f"{args.results_dir}/metrics.csv"
    if os.path.exists(mpath):
        old = pd.read_csv(mpath)
        row = pd.concat([old[old.model != args.label], row], ignore_index=True)
    row.to_csv(mpath, index=False)

    samples = pd.DataFrame({"source": df.src, "reference": refs, "model_output": hyps, "model": args.label}).head(args.n_samples)
    spath = f"{args.results_dir}/sample_translations.csv"
    if os.path.exists(spath):
        old = pd.read_csv(spath)
        samples = pd.concat([old[old.model != args.label], samples], ignore_index=True)
    samples.to_csv(spath, index=False)


if __name__ == "__main__":
    main()
