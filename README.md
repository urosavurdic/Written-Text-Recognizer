# Written Text Recognizer

Handwritten paragraph recognition project built with PyTorch and FastAPI.  
Given an input image of handwriting, it predicts paragraph text using a ResNet encoder + Transformer decoder model.

## Architecture

```text
Input Image
    │
    ▼
ResNet Encoder (visual feature extraction)
    │
    ▼
Transformer Decoder (sequence generation)
    │
    ▼
Text Output
```

## Project Structure

```text
.
├── api_server/                 # FastAPI server for local API inference
├── api_serverless/             # Serverless-friendly FastAPI setup
├── notebooks/                  # Colab / notebook experimentation
├── text_recognizer/            # Core package: data, models, lit models, inference
│   ├── artifacts/              # Trained model artifacts used for inference
│   ├── data/                   # Dataset modules and preprocessing
│   ├── lit_models/             # PyTorch Lightning wrappers
│   ├── models/                 # Neural network architectures
│   └── tests/                  # Recognition tests and sample assets
├── training/                   # Training script and training tests
└── tasks/                      # Lint/test helper scripts
```

## Quick Start (Local Inference)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python text_recognizer/paragraph_text_recognizer.py text_recognizer/tests/support/paragraphs/a01-077.png
```

## API

Run the API server locally:

```bash
python api_server/app.py
```

Example request:

```bash
curl "http://127.0.0.1:8000/v1/predict?image_url=https://fsdl-public-assets.s3-us-west-2.amazonaws.com/paragraphs/a01-077.png"
```

## Training

Training was run in Google Colab due to local hardware limits.  
To reproduce training locally (or in Colab), use the training entrypoint with the paragraph dataset and Transformer model:

```bash
python training/run_training.py --max_epochs 20 --model_class=ResnetTransformer --data_class=IAMParagraphs
```

Artifacts are loaded from `text_recognizer/artifacts/paragraph_text_recognizer/` for inference.

## Results [RESULTS TBD]

| Metric | Value |
|---|---|
| Character Error Rate (CER) | [RESULTS TBD] |
| Validation Loss | [RESULTS TBD] |
| Notes | [RESULTS TBD] |

## Tech Stack

- PyTorch
- PyTorch Lightning
- FastAPI
- Docker

## Acknowledgements

Guided project inspired by the **Full Stack Deep Learning** course (UC Berkeley, 2021):  
https://fullstackdeeplearning.com/
