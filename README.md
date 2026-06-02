# Agentic ML Pipeline

Agentic ML Pipeline is a Streamlit app for end-to-end tabular machine learning. It lets you upload a dataset, inspect and clean features, generate optional LLM-assisted analysis, run feature engineering and selection, apply dimensionality reduction, and train multiple models for either classification or regression.

## What It Does

- Loads CSV and Excel datasets.
- Validates that the dataset has at least two columns.
- Detects the problem type automatically or lets you override it manually.
- Runs dataset analysis and feature-cleaner suggestions.
- Supports optional LLM-based dataset analysis and feature cleaning through OpenRouter.
- Applies preprocessing, feature engineering, feature selection, PCA or ICA, and imbalance handling.
- Trains a model suite and shows a leaderboard with metrics.
- Generates classification diagnostics such as confusion matrix and classification report.
- Lets you download the leaderboard as CSV.

## Tech Stack

- Python 3.12+
- Streamlit
- Pandas and NumPy
- scikit-learn
- imbalanced-learn
- XGBoost
- LightGBM
- OpenAI Python SDK for OpenRouter-compatible requests

## Project Structure

```text
app.py                     # Streamlit UI and pipeline orchestration
agents/
  dataset_analyzer.py      # Dataset analysis and problem-type inference
  feature_cleaner_agent.py # LLM-based feature review and suggestions
  feature_engineer.py      # Rule-based feature engineering helpers
  feature_selector.py      # Feature importance, MI, and RFE selection
  model_trainer.py         # Model training and leaderboard generation
utils/
  data_loader.py           # CSV/Excel loading and validation
  preprocessing.py         # Target encoding and preprocessing pipeline
  imbalance_handler.py     # Class imbalance utilities
  dimensionality_reduction.py # PCA / ICA helpers
  feature_cleaner.py       # Local feature cleanup helpers
  llm.py                   # OpenRouter client setup
models/
  schema.py                # Pydantic schemas used by the pipeline
```

## Installation

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies.

Using `pip`:

```bash
pip install -r requirements.txt
```

Using `uv`:

```bash
uv sync
```

## Configuration

The optional LLM features use OpenRouter.

Create a `.env` file in the project root with:

```env
OPENROUTER_API_KEY=your_api_key_here
```

The default model is defined in [`utils/llm.py`](utils/llm.py). You can change `DEFAULT_MODEL` there if you want to use a different model.

## Run The App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually `http://localhost:8501`.

## How To Use

1. Upload a CSV or Excel file.
2. Choose the target column.
3. Review the dataset analysis and feature-cleaner suggestions.
4. Select the problem type or leave it on auto.
5. Choose optional feature selection, dimensionality reduction, and imbalance handling.
6. Click `Train Models`.
7. Review the leaderboard, best model, and diagnostics.
8. Download the leaderboard if needed.

## Supported Inputs

- `CSV`
- `XLSX`
- `XLS`

The dataset must have at least two columns, and one of them must be a valid target column.

## Notes

- Classification and regression are both supported.
- The app keeps trained model objects in memory for the current session.
- The leaderboard can be downloaded, but models are not currently persisted to disk.
- Some pipelines rely on LLM output, so an invalid or missing API key will disable those parts of the workflow.

## Troubleshooting

- If the app says the dataset is empty or unsupported, check the file format and contents.
- If LLM features fail, confirm that `OPENROUTER_API_KEY` is present in `.env`.
- If model training fails on a tiny dataset, try a larger sample or reduce preprocessing choices that create very small folds.

## License

Add a license here if you want to publish or share the project publicly.
