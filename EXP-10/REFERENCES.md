# References and Third-Party Resources

This project builds on pretrained models, datasets, libraries, and evaluation tools developed by the wider NLP community.

## NLLB-200

The translation model used in this project is:

`facebook/nllb-200-distilled-600M`

NLLB (No Language Left Behind) is a multilingual machine translation project developed by Meta AI.

**Reference**

> NLLB Team et al. *No Language Left Behind: Scaling Human-Centered Machine Translation.* 2022.

Model:
https://huggingface.co/facebook/nllb-200-distilled-600M

The NLLB model is distributed under its own licence and usage conditions. Users should consult the model card before reuse.

---

## Samanantar

The English–Assamese parallel data used for the reported experiment comes from **Samanantar**, developed by AI4Bharat.

**Reference**

> Gowtham Ramesh et al. *Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages.* Transactions of the Association for Computational Linguistics, 2022.

Dataset:
https://huggingface.co/datasets/ai4bharat/samanantar

The dataset is not redistributed in this repository. It is downloaded during preprocessing.

Users should review and comply with the dataset's licence and terms before reuse.

---

## OPUS-100

The preprocessing pipeline also supports OPUS-100 as an alternative data source.

**Reference**

> Biao Zhang et al. *Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation.* ACL, 2020.

Dataset:
https://huggingface.co/datasets/Helsinki-NLP/opus-100

OPUS-100 was **not used for the reported NLLB fine-tuning results**.

---

## SacreBLEU

BLEU and chrF evaluation is performed using SacreBLEU.

**Reference**

> Matt Post. *A Call for Clarity in Reporting BLEU Scores.* WMT, 2018.

Project:
https://github.com/mjpost/sacrebleu

---

## Hugging Face Transformers

Model loading, tokenization, generation, and sequence-to-sequence training use the Hugging Face Transformers ecosystem.

https://github.com/huggingface/transformers

---

## PyTorch

Model training and GPU execution use PyTorch.

https://pytorch.org/

---

## Usage Note

Third-party models, datasets, libraries, and other resources retain their respective licences and terms.

The presence of a reference in this repository does not imply that the repository itself is distributed under the same licence.

At the time this reference file was added, the repository did not contain a project-level `LICENSE` file. Users should not assume a licence for the repository's original source code until the repository owner explicitly provides one.