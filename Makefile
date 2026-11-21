PYTHON ?= python
CONFIG ?= configs/default.yaml

.PHONY: install test dataset train index evaluate all

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	pytest -q

dataset:
	$(PYTHON) scripts/generate_dataset.py --config $(CONFIG)

train:
	$(PYTHON) scripts/train.py --config $(CONFIG)

index:
	$(PYTHON) scripts/build_index.py --config $(CONFIG)

evaluate:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG)

all: dataset train index evaluate
