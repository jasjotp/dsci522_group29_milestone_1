# variables
RAW_URL = https://cran.r-project.org/incoming/UL/diversedata/data-clean/hcmst.csv
RAW_CSV = data/raw/hcmst_raw.csv
VALIDATED_CSV = data/processed/hcmst_validated.csv
PROCESSED_CSV = data/processed/hcmst_processed.csv
PREPROCESSOR = models/preprocessor.joblib
FIG_DIR = figures
RESULT_PREFIX = results/logreg

# default
all: download validate preprocess eda model

# 01 download
download: $(RAW_CSV)

$(RAW_CSV):
	mkdir -p data/raw
	python scripts/01-download-data.py "$(RAW_URL)" $(RAW_CSV)

# 02 validation
validate: $(VALIDATED_CSV)

$(VALIDATED_CSV): $(RAW_CSV)
	mkdir -p data/processed
	python scripts/02-validation.py $(RAW_CSV) $(VALIDATED_CSV)

# 03 preprocessing
preprocess: $(PROCESSED_CSV) $(PREPROCESSOR)

$(PROCESSED_CSV) $(PREPROCESSOR): $(VALIDATED_CSV)
	mkdir -p data/processed models
	python scripts/03-preprocessing.py $(VALIDATED_CSV) $(PROCESSED_CSV) $(PREPROCESSOR)

# 04 EDA
eda: $(FIG_DIR)/dist-relationship-quality.png \
     $(FIG_DIR)/corr_plot.png \
     $(FIG_DIR)/dist-income-category.png

$(FIG_DIR)/dist-relationship-quality.png: $(PROCESSED_CSV)
	mkdir -p $(FIG_DIR)
	python scripts/04-eda.py $(PROCESSED_CSV) $(FIG_DIR)

# 05 model
model: $(RESULT_PREFIX)_metrics.csv

$(RESULT_PREFIX)_metrics.csv: $(PREPROCESSOR) data/processed/X_train.csv data/processed/y_train.csv
	mkdir -p results
	python scripts/05-model.py $(PREPROCESSOR) $(RESULT_PREFIX)

clean:
	rm -rf data/raw/*.csv data/processed/*.csv figures/*.png models/*.joblib results/*

.PHONY: all download validate preprocess eda model clean
