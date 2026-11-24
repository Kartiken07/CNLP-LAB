# Contributing

Thanks for contributing to the AG News Classification project.

This repository contains notebooks, reusable NLP preprocessing and inference code, trained model artifacts, automated tests, and a Flask web application. Changes should preserve consistency between training and inference and keep the project reproducible.

## Development Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/Kartiken07/NLP-PROJECT-1.git
cd NLP-PROJECT-1

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
```

## Development Workflow

1. Start from the latest `main`.
2. Create a focused branch for the change.
3. Make and validate the change locally.
4. Run the automated test suite.
5. Commit with a clear message.
6. Open a pull request describing the change and its validation.

Example branch names:

```text
feature/model-improvement
fix/preprocessing
test/inference
docs/readme-update
chore/reproducibility
```

## Testing

Before opening a pull request, run:

```bash
python -m unittest discover -s tests -v
```

The automated suite covers:

- text preprocessing
- deterministic cleaning
- serialized model inference
- category and confidence outputs
- class probability distributions
- Flask frontend and API routes
- missing, invalid and oversized input handling

All existing tests should pass before changes are merged.

## Training and Inference Consistency

The deployed model depends on the same preprocessing and TF-IDF feature pipeline used during training.

When modifying preprocessing, feature extraction, class mappings, or model artifacts:

- keep training and inference behaviour consistent
- update affected tests
- retrain artifacts when required
- document compatibility changes
- update `MODEL_CARD.md` when model behaviour or performance changes

## Notebooks

The model-development workflow is represented by the project notebooks.

When editing notebooks:

- avoid machine-specific absolute paths
- use repository-relative paths
- keep generated intermediate data out of version control where appropriate
- preserve deterministic random seeds where applicable
- do not commit credentials or local environment files

## Pull Requests

Keep pull requests focused and include:

- a concise summary
- motivation for the change
- affected components
- validation or test results
- updated metrics or figures when model behaviour changes

Avoid combining unrelated model, frontend, documentation, and cleanup changes in a single pull request.

## Model Artifacts

The deployable artifacts are stored in `models/`.

Do not replace trained artifacts without documenting:

- how they were produced
- the relevant dependency versions
- evaluation results
- any changes to preprocessing or feature extraction

## Reporting Problems

Use GitHub Issues for reproducible bugs, inference problems, documentation errors, or proposed improvements.

For prediction-related issues, include a minimal example input and the observed behaviour whenever possible.