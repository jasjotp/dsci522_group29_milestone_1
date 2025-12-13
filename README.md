# DSCI 522: Data Science Workflows – Milestone 1

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
├── data/
│   ├── raw/
│   └── processed/
├── figures/
├── reports/
├── scripts/
├── analysis.ipynb
├── Makefile
├── environment.yml
├── README.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
└── index.html
```

------------------------------------------------------------------------

## 1.How to Run the Analysis

### 1.1. Clone the repository

```
git clone git@github.com:jasjotp/relationship-quality-prediction.git
```

 ```
 cd relationship-quality-prediction
 ```

------------------------------------------------------------------------

## 2. Create the comptational environment

<details>
<summary><b>Option 1: Use `environment.yml`</b></summary>

`conda env create -f environment.yml -n relationship-quality-prediction` <br> `conda activate relationship-quality-prediction`

</details>
------------------------------------------------------------------------

<details>
<summary><b>Option 2: Use `conda-lock.yml`</b></summary>

`conda-lock install --name relationship-quality-prediction conda-lock.yml` <br> `conda activate relationship-quality-prediction`

</details>
------------------------------------------------------------------------

<details>
<summary><b>Option 3: Use `docker-compose.yml`</b></summary>

`docker compose run --rm dsci522_milestone4` <br> `make all`

</details>

After you are done, type `exit` to leave the docker container.

------------------------------------------------------------------------

## 3. Run the Analysis with Make

### Full pipeline

Runs all steps: download, validate, preprocess, generate figures, and train the model.

```
make all
```

### Individual steps

1.  Download raw data 

```
make download
```

2.  Validate the raw data 

```
make validate
```

3.  Preprocess the data and generate train/test splits 

```
make preprocess
```

4.  Generate EDA figures 

```
make eda
```

5.  Train and evaluate model 

```
make model
```

------------------------------------------------------------------------

## 4. Launch the Analysis Notebook

Run

```
jupyter lab
```

Open the `analysis.ipynb` file to run the exploratory analysis and code.

------------------------------------------------------------------------

## 5. Dependencies

All required packages are listed in environment.yml:

Key dependencies include:<br>

```         
<<<<<<< HEAD
  - python=3.12.12
  - numpy=2.3.5
  - pandas=2.3.3
  - matplotlib=3.10.8 
  - seaborn=0.13.2
  - scikit-learn=1.7.2
  - pip=25.3
  - pointblank=0.16.0
  - pandera=0.25.0
  - pytest 9.0.2
  - click=8.3.1
  - ipykernel=7.1.0
  - quarto=1.8.26
=======
python 3.12.12
numpy 2.3.5
pandas 2.3.3
matplotlib 3.10.8 
seaborn 0.13.2
scikit-learn 1.7.2
pip 25.3
pointblank 0.16.0
pandera 0.25.0
pytest 9.0.2
click 8.3.1
ipykernel 7.1.0
quarto 1.8.26
>>>>>>> main
```

To update dependencies, modify `environment.yml` and regenerate the lockfile:

```
conda-lock -f environment.yml --lockfile conda-lock.yml
```

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
