.PHONY: setup preprocess train evaluate clean

# Directories
DATA_DIR = data/BraTS2021_Training_Data
PREPROCESSED_DIR = data/preprocessed
MODELS_DIR = models
RESULTS_DIR = results

# Python environment
PYTHON = python3
PIP = pip3
VENV = venv
VENV_BIN = $(VENV)/bin

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install -r requirements.txt

preprocess:
	$(VENV_BIN)/python scripts/preprocess.py \
		--data-dir $(DATA_DIR) \
		--output-dir $(PREPROCESSED_DIR)

train:
	$(VENV_BIN)/python scripts/train.py \
		--data-dir $(PREPROCESSED_DIR) \
		--output-dir $(MODELS_DIR) \
		--epochs 100

evaluate:
	$(VENV_BIN)/python scripts/evaluate.py \
		--data-dir $(PREPROCESSED_DIR) \
		--model-path $(MODELS_DIR)/model_best.pth \
		--output-dir $(RESULTS_DIR)

clean:
	rm -rf $(PREPROCESSED_DIR)
	rm -rf $(MODELS_DIR)
	rm -rf $(RESULTS_DIR)
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name "*.pyc" -exec rm -r {} + 