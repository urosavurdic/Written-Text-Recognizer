# Written Text Recognizer

An end-to-end handwritten text recognition project that takes an image of text (line or paragraph) and predicts machine-readable text using deep learning.

## Architecture

```text
Input Image
    ↓
ResNet Encoder (extract visual features)
    ↓
Transformer Decoder (generate character sequence)
    ↓
Text Output
```

## Project Structure

```text
.
├── api_server/                 # FastAPI app for local API inference
├── api_serverless/             # Dockerized serverless-style API setup
├── text_recognizer/            # Core package: data, models, lit models, inference
│   ├── data/                   # Data modules (IAM, EMNIST, synthetic, fake data for tests)
│   ├── models/                 # Neural network architectures (CNN, LSTM, Transformer, etc.)
│   ├── lit_models/             # PyTorch Lightning training wrappers
│   └── tests/                  # Unit/integration tests for core behavior
├── training/                   # Training entrypoint + training tests
└── tasks/                      # Lint/test helper scripts
```

## Quick Start

```bash
git clone https://github.com/urosavurdic/Written-Text-Recognizer.git
cd Written-Text-Recognizer
pip install -r requirements.txt
PYTHONPATH=. python text_recognizer/paragraph_text_recognizer.py text_recognizer/tests/support/paragraphs/a01-077.png
```

## API

Run the local API server:

```bash
PYTHONPATH=. python api_server/app.py
```

Example request:

```bash
curl -X GET "http://127.0.0.1:8000/v1/predict?image_url=https://fsdl-public-assets.s3-us-west-2.amazonaws.com/paragraphs/a01-077.png"
```

## Training

Model training and experimentation were primarily run in Google Colab (GPU access), then checkpoints were used locally for inference and API serving.

To reproduce a lightweight training run locally:

```bash
PYTHONPATH=. python training/run_training.py --data_class=FakeImageData --model_class=CNN --conv_dim=32 --fc_dim=16 --loss=cross_entropy --num_workers=4 --max_epochs=4
```

## Results

`TODO`

| Model | Dataset | Metric | Score | Notes |
|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO |

## Tech Stack

- PyTorch
- PyTorch Lightning
- FastAPI
- Docker

## Acknowledgements

This project was built as part of the Full Stack Deep Learning course (UC Berkeley, 2021):  
https://fullstackdeeplearning.com/
