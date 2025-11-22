*DSCI 522: Data Science Workflows – Milestone 1*
---
## TABLE OF CONTENTS<br>

1.  **Project Title**
2.  **Authors**
3.  **Summary**
4.  **How to run analysis**
5.  **Dependencies**
6.  **Contributing**
7.  **Licenses**
8.  **Acknowledgements**

---

## Project Title 
**PROJECT HCMST a.k.a. "How Couples Meet and Stay Together**"

## 👥 Authors
- Eugene Tse  
- Jade Chen  
- Jasjot Parmar  
- Johnson Leung  

---

## Project Summary 📄

This project analyzes data from the **How Couples Meet and Stay Together (HCMST)** survey to explore how demographic and relationship characteristics relate to self-reported **relationship quality**. We focus on a subset of variables including respondent age, income category, marital status, relationship duration, and number of children.

We begin with **exploratory data analysis (EDA)**, examining the distribution of relationship quality, visualizing the income distribution, and computing a correlation matrix for key predictors and the outcome. We then build a **multi-class logistic regression model** to predict relationship quality from these features. The workflow includes explicit data cleaning (handling missing values, type conversion), appropriate encoding of ordinal income categories, and scaling of numeric predictors via a scikit-learn preprocessing pipeline.

Model performance is evaluated using **confusion matrices** on both training and test data, as well as a **micro-averaged one-vs-rest ROC curve** to assess overall discriminative ability across classes. Together, the analysis and models provide an interpretable baseline for understanding which characteristics are most associated with higher or lower reported relationship quality, and they establish a reproducible workflow that can be extended with more complex models in later milestones.

---

## Repository Structure 📁
```
├── analysis.ipynb
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── data/
      └── hcmst.csv
├── environment.yml
├── conda-lock.yml
├── LICENSE
└── README.md
```
---

##  1.How to Run the Analysis 

### 1.1. Clone the repository

`git clone https://github.com/jasjotp/dsci522_group29_milestone_1.git` <br>
`cd dsci522_group29_milestone_1`

---

##  2. Create the conda environment

`conda env create -f environment.yml -n dsci522_milestone1` <br>
`conda activate dsci522_milestone1`

---

## 3. Use the Lockfile

For reproducible installs:

`conda-lock install -n dsci522_milestone1 conda-lock.yml` <br>
`conda activate dsci522_milestone1`

---

## 4. Launch the Analysis Notebook

Run<br>

`jupyter lab`<br>

Open the `analysis.ipynb` file to run the exploratory analysis and code.

---

## 5. Dependencies

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

---

## 6. Contributing

We welcome contributions!<br>

Please review:

-  `CONTRIBUTING.md` – contribution workflow, coding guidelines
-  `CODE_OF_CONDUCT.md` – community standards and expected behaviour

All contributing members are expected to follow these documents when contributing.

---

## 7. License

This project is covered under a `MIT License`, as declared in the project root.

By contributing to this repository, you agree that your contributions will be covered under the same license.

---

## 8. Acknowledgements
This project was created as part of the Master of Data Science (MDS) program at the University of British Columbia.
