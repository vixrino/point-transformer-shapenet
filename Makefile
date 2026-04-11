VENV    := .venv
PYTHON  := $(VENV)/bin/python3
PIP     := $(VENV)/bin/pip
DATA    ?= ./data/ShapeNet
EPOCHS  ?= 10
CATS    ?= Airplane Chair Car Table Lamp
K       ?= 8
BATCH   ?= 16

SYSPYTHON := $(shell command -v python3.11 || command -v python3.12 || command -v python3)

$(VENV)/bin/activate:
	$(SYSPYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

.PHONY: install train train-resume seg seg-resume sanity-check eval clean

install: $(VENV)/bin/activate
	$(PIP) install -r requirements.txt

download: $(VENV)/bin/activate
	$(PYTHON) download_shapenet.py --data_root $(DATA)

train: $(VENV)/bin/activate download
	$(PYTHON) train.py \
		--data_root $(DATA) \
		--epochs $(EPOCHS) \
		--categories $(CATS) \
		--k $(K) \
		--batch_size $(BATCH)

train-resume: $(VENV)/bin/activate download
	$(PYTHON) train.py \
		--data_root $(DATA) \
		--epochs $(EPOCHS) \
		--categories $(CATS) \
		--k $(K) \
		--batch_size $(BATCH) \
		--resume

seg: $(VENV)/bin/activate download
	$(PYTHON) train_segmentation.py \
		--data_root $(DATA) \
		--epochs 15 \
		--k $(K) \
		--batch_size $(BATCH)

seg-resume: $(VENV)/bin/activate download
	$(PYTHON) train_segmentation.py \
		--data_root $(DATA) \
		--epochs 15 \
		--k $(K) \
		--batch_size $(BATCH) \
		--resume

sanity-check: $(VENV)/bin/activate
	$(PYTHON) sanity_check.py --shape chair --save sanity_chair_backrest.png
	$(PYTHON) sanity_check.py --shape chair --y_min -0.1 --y_max 0.38 --save sanity_chair_legs.png
	$(PYTHON) sanity_check.py --shape table --y_min 0.35 --y_max 1.0 --save sanity_table_top.png
	@echo "Images sauvegardées : sanity_chair_backrest.png, sanity_chair_legs.png, sanity_table_top.png"

eval: $(VENV)/bin/activate
	$(PYTHON) train.py \
		--data_root $(DATA) \
		--epochs 0 \
		--categories $(CATS)

clean:
	rm -rf $(VENV) __pycache__ models/__pycache__ data/
	rm -f *.aux *.log *.out *.toc report_*.tex report_*.pdf
