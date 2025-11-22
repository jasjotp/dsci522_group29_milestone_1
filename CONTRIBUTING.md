# Contributing to PROJECT-HCMST
First off, thank you for your contribution! 🎉  
<br>
<br>
This document explains how to set up your environment, propose changes, and work with the rest of the team on this project.

The project is currently maintained by:

- Eugene Tse  :metal:
- Jade Chen  :nail_care:
- Jasjot Parmar  :muscle:
- Johnson Leung  :punch:

---

## 1. Code of Conduct

By participating in this project, you agree to abide by the project’s Code of Conduct (`CODE_OF_CONDUCT.md`).

In short:

- **Be respectful and inclusive** – no harassment, discrimination, or exclusionary behaviour.
- **Assume good intent** and resolve disagreements with empathy and professionalism.
- **Collaborate openly** – share context early, involve relevant teammates, and communicate clearly.
- **Ask for help early** and answer questions in a supportive way.
- **Step down considerately** – if you need to reduce your involvement, please communicate and help with handover.

Serious or repeated violations (e.g., harassment, hostile communication, personal attacks) may lead to warnings, removal from specific tasks, or escalation to the instructor/TA. 

Please read the full `CODE_OF_CONDUCT.md` for details before contributing. 

---

## 2. Project Overview

This repository contains the work for **PROJECT HCMST**, a group project for UBC'S MDS - DSCI 522.  
The `README.md` includes information for:

- **How Couples Meet Stay Together (HCMST)**
- **Summary**
- **How-to run analysis**
- **Dependencies**
- **Licenses** (MIT License)

These sections should be updated and kept in sync with any changes you make to the project. 

---

## 3. How to Set Up Your Environment

We use **conda** for environment management, with both an `environment.yml` and a generated `conda-lock.yml`.

### 3.1. Using `environment.yml` (recommended)

To create a development environment from the repository root:

`conda env create -f environment.yml -n dsci522_milestone1` <br>
`conda activate dsci522_milestone1`

The base environment includes versions of:<br>
Python 3.12.12<br>
numpy 2.3.5<br>
pandas 2.3.3<br>
matplotlib 3.10.8<br>
seaborn 0.13.2<br>
altair 6.0.0<br>
scikit-learn 1.7.2<br>
conda 25.7.0 <br>
environment<br>

### 3.2. Using `conda-lock.yml` (for fully reproducible installs)
`conda-lock.yml` is an auto-generated lock file that pins exact package versions (including checksums) for multiple platforms. Do not edit it by hand. <br>

To install the exact locked environment:

`conda-lock install -n dsci522_milestone1 conda-lock.yml`
`conda activate dsci522_milestone1`

Key notes:
-  The lock file is generated from `environment.yml`.
-  To update a single package (e.g., altair) while respecting existing constraints:
  
`conda-lock lock --lockfile conda-lock.yml --update altair`

To fully re-solve the environment after changing environment.yml:

`conda-lock -f environment.yml --lockfile conda-lock.yml`

If you change dependencies, you must:

1.  Update `environment.yml`.
2.  Regenerate `conda-lock.yml` using conda-lock as shown above.
3.  Commit both files.

---

## 4. Workflow: How to Contribute Changes
### 4.1. Typical contribution flow
#### 4.1.1. Create a branch

`git checkout -b feature/short-description`

Examples:<br>
`feature/add-eda-notebook`<br>
`bugfix/fix-altair-plot`<br>
`docs/update-readme`<br>

####  4.1.2. Make your changes
-  Follow the environment setup above.
-  Keep notebooks, scripts, and docs organized.
-  Prefer small, focused commits with clear messages.

#### 4.1.3. Run checks (if available)
-  Run any tests or analysis scripts the team agrees on (e.g., pytest, make test, or a specific python/bash script).
-  Ensure your code runs end-to-end in the project environment.

#### 4.1.4. Update documentation
-  Update `README.md` when:
    -  The way to run the analysis changes.
    -  Dependencies change.
    -  The project title or summary becomes more concrete.
-  Update `CONTRIBUTING.md` if you change the contribution process.
-  If relevant, add comments/markdown cells to notebooks explaining your changes.

#### 4.1.5.  Open a pull request (PR)
-  Summarize the motivation and high-level changes.
-  Link to any related issues/tasks.
-  Mention any breaking changes or new dependencies.
-  Ask for at least one review from a teammate.

#### 4.1.6.  Address review feedback
-  Respond to comments respectfully.
-  Push follow-up commits rather than force-pushing (unless the team agrees otherwise).
-  Ensure final CI/tests pass (if configured).

---

## 5. Working with Notebooks and Data
-  Keep notebooks clean:
    -  Restart kernel and run all cells before committing.
    -  Avoid committing large, unused intermediate outputs.
-  Data handling:
    -  Do not commit large or sensitive data files directly unless explicitly allowed.
    -  Prefer scripts/notebooks that download or generate data from documented sources.

If you add or change data sources, update the `README.md` and any relevant documentation so that others can reproduce your analysis. 

---

## 6. Coding & Style Guidelines
   
These are suggested defaults; align within the group and adjust as needed:
-  Python
    -  Follow PEP 8 style where feasible.
    -  Use clear, descriptive variable and function names.
    -  Prefer functions and small modules over monolithic scripts.
-  R / notebooks 
    -  Keep code chunks focused and documented.
    -  Use meaningful object names and comments for non-obvious logic.
-  Visualization
    -  Use informative axis labels, legends, and titles.
    -  Make plots reproducible (no manual click-driven steps).
-  Reproducibility
    -  Avoid hard-coding local file paths.
    -  Use relative paths within the repo (e.g., data/, results/, figures/).

---

## 7. Updating Dependencies
   
When you need a new Python package or version bump:

-  Add/update it in environment.yml under dependencies. <br>
-  Re-lock the environment:

`conda-lock -f environment.yml --lockfile conda-lock.yml`

-  Confirm the project still runs in a fresh environment.<br>
-  Commit both environment.yml and conda-lock.yml and mention the dependency change in your PR.

---

## 8. Reporting Issues & Requesting Help

If you encounter a problem (bug, unclear documentation, environment issues, etc.):
-  Check existing issues/PRs (if using GitHub Issues).
-  When opening a new issue, please include:
    -  A short, descriptive title.
    -  Steps to reproduce (commands, environment info).
    -  Expected vs. actual behavior.
    -  Error messages or screenshots where helpful.

Within the team, you’re encouraged to:
-  Ask questions early to avoid getting blocked.
-  Answer others’ questions kindly and constructively.
-  Reference relevant parts of the Code of Conduct in case of conflict. 

---

## 9. License & Attribution
    
The project is currently marked as using a MIT License in the README. 

By contributing, you agree that your contributions may be distributed under the same license as the project.

If you add third-party code or resources:

-  Ensure their license is compatible.
-  Attribute the source appropriately in the code, `README.md`.

---

## 10. Questions?
If anything in this document is unclear or out of date:
Open an issue or PR to improve `CONTRIBUTING.md`, or
Discuss with your teammates and update the docs so future contributors benefit, if applicable.


Thanks again for contributing.

Logging off,


TEAM 29MDS
💜






