# variables
RAW_URL = https://cran.r-project.org/incoming/UL/diversedata/data-clean/hcmst.csv

RAW_DIR     = data/raw
PROC_DIR    = data/processed
MODEL_DIR   = models
FIG_DIR     = figures
RESULTS_DIR = results
PLOTS_DIR   = $(RESULTS_DIR)/plots
REPORT_DIR  = reports

RAW_CSV       = $(RAW_DIR)/hcmst_raw.csv
VALIDATED_CSV = $(PROC_DIR)/hcmst_validated.csv

X_TRAIN = $(PROC_DIR)/X_train.csv
X_TEST  = $(PROC_DIR)/X_test.csv
Y_TRAIN = $(PROC_DIR)/y_train.csv
Y_TEST  = $(PROC_DIR)/y_test.csv

PREPROCESSOR = $(MODEL_DIR)/preprocessor.joblib
LOGREG_MODEL = $(RESULTS_DIR)/logreg.pkl
DUMMY_MODEL  = $(RESULTS_DIR)/dummy_model.pkl
ROC_PLOT     = $(PLOTS_DIR)/roc_curve.pkl

EDA_FIGURES = \
	$(FIG_DIR)/dist-relationship-quality.png \
	$(FIG_DIR)/corr_plot.png \
	$(FIG_DIR)/dist-income-category.png

REPORT_QMD  = $(REPORT_DIR)/relationship_quality_analysis.qmd
REPORT_HTML = $(REPORT_DIR)/relationship_quality_analysis.html
REPORT_PDF  = $(REPORT_DIR)/relationship_quality_analysis.pdf


# default target
all: download validate preprocess eda model report


# 01 download
download: $(RAW_CSV)

$(RAW_CSV):
	mkdir -p $(RAW_DIR)
	python scripts/01-download-data.py "$(RAW_URL)" $(RAW_CSV)


# 02 validation
validate: $(VALIDATED_CSV)

$(VALIDATED_CSV): $(RAW_CSV)
	mkdir -p $(PROC_DIR)
	python scripts/02-validation.py $(RAW_CSV) $(VALIDATED_CSV)


# 03 preprocessing and splitting
preprocess: $(X_TRAIN) $(X_TEST) $(Y_TRAIN) $(Y_TEST) $(PREPROCESSOR)

$(X_TRAIN) $(X_TEST) $(Y_TRAIN) $(Y_TEST) $(PREPROCESSOR): $(VALIDATED_CSV)
	mkdir -p $(PROC_DIR) $(MODEL_DIR)
	python scripts/03-preprocessing.py $(VALIDATED_CSV) $(PROC_DIR) $(PREPROCESSOR)


# 04 EDA
eda: $(EDA_FIGURES)

$(EDA_FIGURES): $(VALIDATED_CSV)
	mkdir -p $(FIG_DIR)
	python scripts/04-eda.py $(VALIDATED_CSV) $(FIG_DIR)


# 05 model
model: $(LOGREG_MODEL) $(DUMMY_MODEL) $(ROC_PLOT)

$(LOGREG_MODEL) $(DUMMY_MODEL) $(ROC_PLOT): $(X_TRAIN) $(X_TEST) $(Y_TRAIN) $(Y_TEST) $(PREPROCESSOR)
	mkdir -p $(RESULTS_DIR) $(PLOTS_DIR)
	python scripts/05-model.py $(PROC_DIR) $(PREPROCESSOR) $(LOGREG_MODEL)


# 06 report
report: $(REPORT_HTML) $(REPORT_PDF)

$(REPORT_HTML): $(REPORT_QMD) $(EDA_FIGURES) $(LOGREG_MODEL) $(DUMMY_MODEL)
	quarto render $(REPORT_QMD) --to html

$(REPORT_PDF): $(REPORT_QMD) $(EDA_FIGURES) $(LOGREG_MODEL) $(DUMMY_MODEL)
	quarto render $(REPORT_QMD) --to pdf


# clean
clean:
	rm -rf data/raw data/processed figures models results reports/*.html reports/*.pdf


.PHONY: all clean download validate preprocess eda model report