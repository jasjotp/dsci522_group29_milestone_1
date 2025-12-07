*DSCI 522: Data Science Workflows – Milestone 1* 31b8e172-b470-440e-83d8-e6b185028602:dAB5AHAAZQA6AFoAUQBBAHgAQQBEAGcAQQBNAFEAQQA1AEEARABZAEEATQBBAEEAMQBBAEMAMABBAE0AQQBCAGgAQQBHAE0AQQBaAEEAQQB0AEEARABRAEEAWgBnAEIAaABBAEcAVQBBAEwAUQBBADQAQQBHAEkAQQBPAFEAQQA1AEEAQwAwAEEATwBRAEIAagBBAEQARQBBAFkAZwBBADMAQQBHAEkAQQBaAEEAQQAzAEEARwBNAEEATQBBAEIAbQBBAEQARQBBAAoAcABvAHMAaQB0AGkAbwBuADoATgBBAEEANQBBAEEAPQA9AAoAcAByAGUAZgBpAHgAOgAKAHMAbwB1AHIAYwBlADoATABRAEEAdABBAEMAMABBAEMAZwBBAGoAQQBDAE0AQQBJAEEAQgBVAEEARQBFAEEAUQBnAEIATQBBAEUAVQBBAEkAQQBCAFAAQQBFAFkAQQBJAEEAQgBEAEEARQA4AEEAVABnAEIAVQBBAEUAVQBBAFQAZwBCAFUAQQBGAE0AQQBQAEEAQgBpAEEASABJAEEAUABnAEEASwBBAEEAbwBBAE0AUQBBAHUAQQBDAEEAQQBJAEEAQQBxAEEAQwBvAEEAVQBBAEIAeQBBAEcAOABBAGEAZwBCAGwAQQBHAE0AQQBkAEEAQQBnAEEARgBRAEEAYQBRAEIAMABBAEcAdwBBAFoAUQBBAHEAQQBDAG8AQQBDAGcAQQB5AEEAQwA0AEEASQBBAEEAZwBBAEMAbwBBAEsAZwBCAEIAQQBIAFUAQQBkAEEAQgBvAEEARwA4AEEAYwBnAEIAegBBAEMAbwBBAEsAZwBBAEsAQQBEAE0AQQBMAGcAQQBnAEEAQwBBAEEASwBnAEEAcQBBAEYATQBBAGQAUQBCAHQAQQBHADAAQQBZAFEAQgB5AEEASABrAEEASwBnAEEAcQBBAEEAbwBBAE4AQQBBAHUAQQBDAEEAQQBJAEEAQQBxAEEAQwBvAEEAUwBBAEIAdgBBAEgAYwBBAEkAQQBCADAAQQBHADgAQQBJAEEAQgB5AEEASABVAEEAYgBnAEEAZwBBAEcARQBBAGIAZwBCAGgAQQBHAHcAQQBlAFEAQgB6AEEARwBrAEEAYwB3AEEAcQBBAEMAbwBBAEMAZwBBADEAQQBDADQAQQBJAEEAQQBnAEEAQwBvAEEASwBnAEIARQBBAEcAVQBBAGMAQQBCAGwAQQBHADQAQQBaAEEAQgBsAEEARwA0AEEAWQB3AEIAcABBAEcAVQBBAGMAdwBBAHEAQQBDAG8AQQBDAGcAQQAyAEEAQwA0AEEASQBBAEEAZwBBAEMAbwBBAEsAZwBCAEQAQQBHADgAQQBiAGcAQgAwAEEASABJAEEAYQBRAEIAaQBBAEgAVQBBAGQAQQBCAHAAQQBHADQAQQBaAHcAQQBxAEEAQwBvAEEAQwBnAEEAMwBBAEMANABBAEkAQQBBAGcAQQBDAG8AQQBLAGcAQgBNAEEARwBrAEEAWQB3AEIAbABBAEcANABBAGMAdwBCAGwAQQBIAE0AQQBLAGcAQQBxAEEAQQBvAEEATwBBAEEAdQBBAEMAQQBBAEkAQQBBAHEAQQBDAG8AQQBRAFEAQgBqAEEARwBzAEEAYgBnAEIAdgBBAEgAYwBBAGIAQQBCAGwAQQBHAFEAQQBaAHcAQgBsAEEARwAwAEEAWgBRAEIAdQBBAEgAUQBBAGMAdwBBAHEAQQBDAG8AQQBDAGcAQQBLAEEAQwAwAEEATABRAEEAdABBAEEAPQA9AAoAcwB1AGYAZgBpAHgAOgA=:31b8e172-b470-440e-83d8-e6b185028602

## Project Title

**PROJECT HCMST a.k.a. "How Couples Meet and Stay Together**"

## 👥 Authors - TEAM 29MDS 💜

-   Eugene Tse 🤘
-   Jade Chen 💅
-   Jasjot Parmar 💪
-   Johnson Leung 👊

------------------------------------------------------------------------

## Project Summary 📄

This project analyzes data from the **How Couples Meet and Stay Together (HCMST)** survey to explore how demographic and relationship characteristics relate to self-reported **relationship quality**. We focus on a subset of variables including respondent age, income category, marital status, relationship duration, and number of children.

We begin with **exploratory data analysis (EDA)**, examining the distribution of relationship quality, visualizing the income distribution, and computing a correlation matrix for key predictors and the outcome. We then build a **multi-class logistic regression model** to predict relationship quality from these features. The workflow includes explicit data cleaning (handling missing values, type conversion), appropriate encoding of ordinal income categories, and scaling of numeric predictors via a scikit-learn preprocessing pipeline.

Model performance is evaluated using **confusion matrices** on both training and test data, as well as a **micro-averaged one-vs-rest ROC curve** to assess overall discriminative ability across classes. Together, the analysis and models provide an interpretable baseline for understanding which characteristics are most associated with higher or lower reported relationship quality, and they establish a reproducible workflow that can be extended with more complex models in later milestones.

View the rendered analysis here: https://jasjotp.github.io/relationship-quality-prediction/

------------------------------------------------------------------------

## Repository Structure 📁

```         
├── analysis.ipynb
├── index.html
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── data/
      └── hcmst.csv
├── environment.yml
├── conda-lock.yml
├── LICENSE
└── README.md
```

------------------------------------------------------------------------

## 1.How to Run the Analysis

### 1.1. Clone the repository

`git clone git@github.com:jasjotp/relationship-quality-prediction.git` <br> `cd relationship-quality-prediction`

------------------------------------------------------------------------

## 2. Create the conda environment

`conda env create -f environment.yml -n relationship-quality-prediction` <br> `conda activate relationship-quality-prediction`

------------------------------------------------------------------------

## 3. Use the Lockfile

For reproducible installs:

`conda-lock install -n relationship-quality-prediction conda-lock.yml` <br> `conda activate relationship-quality-prediction`

------------------------------------------------------------------------

## 4. Run the Analysis with Make

### Full pipeline

Runs all steps: download, validate, preprocess, generate figures, and train the model.

`make all`

### Individual steps

1.  Download raw data `make download`
2.  Validate the raw data `make validate`
3.  Preprocess the data and generate train/test splits `make preprocess`
4.  Generate EDA figures `make eda`
5.  Train and evaluate model `make model`

------------------------------------------------------------------------

## 5. Launch the Analysis Notebook

Run<br>

`jupyter lab`<br>

Open the `analysis.ipynb` file to run the exploratory analysis and code.

------------------------------------------------------------------------

## 6. Dependencies

All required packages are listed in environment.yml:

Key dependencies include:<br>

```         
Python 3.12.12
numpy 2.3.5
pandas 2.3.3
matplotlib 3.10.0
seaborn 0.13.2
altair 6.0.0
scikit-learn 1.7.2
conda 25.7
```

To update dependencies, modify `environment.yml` and regenerate the lockfile:

`conda-lock -f environment.yml --lockfile conda-lock.yml`

------------------------------------------------------------------------

## 6. Contributing

We welcome contributions!<br>

Please review:

-   `CONTRIBUTING.md` – contribution workflow, coding guidelines
-   `CODE_OF_CONDUCT.md` – community standards and expected behaviour

All contributing members are expected to follow these documents when contributing.

------------------------------------------------------------------------

## 7. License

This project is covered under a `MIT License`, as declared in the project root.

By contributing to this repository, you agree that your contributions will be covered under the same license.

------------------------------------------------------------------------

## 8. Acknowledgements

This project was created as part of the Master of Data Science (MDS) program at the University of British Columbia.