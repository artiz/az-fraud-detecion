# Local workflow. Requires the conda env from environment.yml (micromamba create -f environment.yml).
PY ?= python
DATA_DIR ?= data/processed

.PHONY: env data baseline gnn evaluate explain serve test lint cfn-lint plot clean

env:
	micromamba create -y -f environment.yml
	@echo "activate with: micromamba activate graph-fraud"

data:
	$(PY) scripts/prepare_data.py --raw-dir data/raw --out-dir $(DATA_DIR)

baseline:
	$(PY) scripts/train.py --model xgboost   --data-dir $(DATA_DIR) --model-dir artifacts/xgboost

gnn:
	$(PY) scripts/train.py --model graphsage --data-dir $(DATA_DIR) --model-dir artifacts/graphsage

evaluate:
	$(PY) scripts/evaluate.py --model-dir artifacts/xgboost   --data-dir $(DATA_DIR) --out-dir artifacts/xgboost/eval
	$(PY) scripts/evaluate.py --model-dir artifacts/graphsage --data-dir $(DATA_DIR) --out-dir artifacts/graphsage/eval

explain:
	$(PY) scripts/explain.py --model-dir artifacts/xgboost   --data-dir $(DATA_DIR) --out artifacts/xgboost/explanations.json
	$(PY) scripts/explain.py --model-dir artifacts/graphsage --data-dir $(DATA_DIR) --out artifacts/graphsage/explanations.json

serve:
	MODEL_DIR=artifacts/graphsage $(PY) scripts/serve.py

plot:
	$(PY) scripts/plot_results.py --out docs/f1_per_timestep.png

test:
	$(PY) -m pytest

lint:
	ruff check . && ruff format --check .

cfn-lint:
	cfn-lint cloudformation/*.yaml

clean:
	rm -rf artifacts data/processed .pytest_cache .ruff_cache
