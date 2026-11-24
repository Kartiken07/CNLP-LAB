# Contributing

Contributions that improve reproducibility, documentation, evaluation, or the English–Assamese translation pipeline are welcome.

## Development Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Kartiken07/LOW-RESOURCE-NEURAL-MACHINE-TRANSLATION.git
cd LOW-RESOURCE-NEURAL-MACHINE-TRANSLATION

pip install -r requirements.txt
```

GPU access is recommended for model training and evaluation.

## Making a Contribution

1. Fork the repository.
2. Create a focused branch from the latest `main`.
3. Make and test your changes.
4. Keep generated datasets and model checkpoints out of Git.
5. Commit with a clear description of the change.
6. Open a pull request explaining what was changed and how it was validated.

## Validation

For changes to preprocessing or repository infrastructure, run:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

For model-training or evaluation changes, document:

- dataset and preprocessing configuration
- base model
- translation direction
- training hyperparameters
- random seed
- hardware
- training time
- number of evaluation examples
- BLEU and chrF
- qualitative examples or error analysis

Do not report estimated model scores as experimental results.

## Large Files

Do not commit:

- downloaded datasets
- trained model directories
- checkpoints
- `.safetensors`, `.bin`, or `.pt` model files
- archives containing model artifacts

Use Kaggle, Hugging Face, or another appropriate external storage service for large trained artifacts.

## Secrets

Never commit:

- API keys
- access tokens
- passwords
- private dataset credentials
- `.env` files

If a secret is accidentally committed, revoke it immediately and remove it from repository history.

## Scope

Keep pull requests focused.

Examples of useful contributions include:

- preprocessing improvements
- reproducibility fixes
- evaluation improvements
- tests
- notebook reliability
- documentation
- carefully controlled English–Assamese experiments

Changes involving additional models, language pairs, or substantially different experimental protocols should clearly explain how they relate to the project's original scope.

## Third-Party Resources

This project uses external models and datasets with their own licences and terms.

Contributors are responsible for complying with the licences and usage restrictions of any external resources they introduce.