# Changelog

All changes made to this project after receiving peer and TA feedback are documented below.

### Addressed Peer Feedback

#### Improved citation handling and references formatting
- **Feedback:** Peer reviewer said to convert hard-coded URLs in the report into formal references using APA-style citations and the Quarto references block using (`::: {#refs}`).
- **Change made:** 
  - Added a references block at the bottom of the report using `::: {#refs}`.
  - Converted URLs related to the HCMST dataset into actual BibTeX citations.
  - Updated narrative text to cite sources using `@hcmst_data_stanford` instead of raw URLs.
- **Evidence:** 
  - See report file `reports/relationship_quality_analysis.qmd`, References section (lines 580-581).
  - GitHub commit id: 3d8b65e72348fb4703d6ced3466ed9f57bc95314
  - GitHub commit link: https://github.com/jasjotp/relationship-quality-prediction/commit/3d8b65e72348fb4703d6ced3466ed9f57bc95314

  - Inline citations updated in the data description section (lines 47) in the Introduction section.
  - Commit ID: f2abc298857581ccd7bfe839ee27d701f0f51d5b
  - Commit Link (6th file is the report): https://github.com/jasjotp/relationship-quality-prediction/commit/f2abc298857581ccd7bfe839ee27d701f0f51d5b 
  - File changed: `reports/relationship_quality_analysis.qmd`

---

### Addressed TA Feedback

#### Added version requirements for dependencies in README
- **Feedback:** README had dependencies but did not specify version requirements.
- **Change made:** Added versione for all dependencies in `README.md`.
- **Evidence:** 
  - See `README.md`, Dependencies section (lines 169 - 183).
  - Pinned every version to each dependency.
  - Commit ID: 9d6bc3e2319fe14b3bce5704bf6f3511c08e452c
  - Commit Link: https://github.com/jasjotp/relationship-quality-prediction/commit/9d6bc3e2319fe14b3bce5704bf6f3511c08e452c

#### Renamed analysis notebook to be descriptive
- **Feedback:** Analysis file was named `analysis.ipynb`, which is not as descriptive as needed. 
- **Change made:** Renamed the analysis notebook to a descriptive filename reflecting the project purpose.
- **Evidence:** 
  - File renamed from `analysis.ipynb` to `relationship_quality_analysis.ipynb`. On very bottom of the commit page. 
  - Commit ID: d249167c7758bc9b23cdc27b95518be1b9544611
  - Commit link: https://github.com/jasjotp/relationship-quality-prediction/commit/d249167c7758bc9b23cdc27b95518be1b9544611#diff-9b2f22ed02646d6da5d5ed4fe45c760fc3d9921acb1260eab8a4a61b1c929698

#### Referenced citations in the narrative text
- **Feedback:** References were at the end of the report but not cited in the narrative. 
- **Change made:** Added in-text citations in the report where relevant so all references show in the references list at the bottom.
- **Evidence:** 
  - See Introduction sections of `relationship_quality_analysis.qmd`. (lines/block 47)
- Commit ID: f2abc298857581ccd7bfe839ee27d701f0f51d5b
- Commit Link (6th file is the report): https://github.com/jasjotp/relationship-quality-prediction/commit/f2abc298857581ccd7bfe839ee27d701f0f51d5b 

#### Improved Docker image by specifiying a version for docker-compose and only adding needed volumes. 
- **Feedback:** 
  - Docker image did not have a version tag.
  - docker-compose mounted all volumnes instead of only necessary directories.
- **Change made:** 
  - Added a version tag to the Docker image.
  - Updated volume mounts to only include the required directories for analysis.
- **Evidence:** 
  - See `docker-compose.yml` changes (lines 2 and lines 8-15).
  - Commit ID: d249167c7758bc9b23cdc27b95518be1b9544611
  - Commit Link: https://github.com/jasjotp/relationship-quality-prediction/commit/d249167c7758bc9b23cdc27b95518be1b9544611#diff-9b2f22ed02646d6da5d5ed4fe45c760fc3d9921acb1260eab8a4a61b1c929698
 
---

GENE-008's (Eugene Tse) changes

Addressed TA and Peer comments (see GitHub Issues -> https://github.com/UBC-MDS/data-analysis-review-2025/issues/35)
- dec966c5e47ce802f20a29ae6a02d35192a80acb

Added Dummy variable (as per Peer Christine requested )
- d46bbd5d996dab91aa3a0b3e32982f941c09ed77

---

#### Implemented full Makefile workflow

* **Feedback:** Project needed a root-level Makefile with `all` and `clean` targets, proper dependencies, and documentation.
* **Change made:**

  * Added a complete Makefile to run the pipeline end to end.
  * Implemented `.PHONY` `all` and `clean` targets.
  * Refactored targets to use dependency-based execution.
  * Standardized paths to use the `data/processed` workflow.
  * Suppressed Pandera import warnings, fixed histogram tick labels, ignored `__pycache__`, and regenerated all outputs.
* **Evidence:**

  * Pull request: #54
  * Related commits include Makefile refactors, preprocessing cleanup, plot fixes, and full pipeline regeneration.

#### Added preprocessing test

* **Feedback:** Preprocessing logic required test coverage.
* **Change made:** Added a preprocessing unit test. Abstraction was handled by a teammate.
* **Evidence:** Pull request: #56

#### Revised report formatting and content

* **Feedback:** Report was missing HTML authors, affiliations, figure numbers, and had grammar issues.
* **Change made:**

  * Added authors and affiliations.
  * Added figure numbers and labels.
  * Fixed grammar and spelling issues.
* **Evidence:** Pull request: #58

#### Added contact email to Code of Conduct

* **Feedback:** `CODE_OF_CONDUCT.md` lacked a contact email.
* **Change made:** Added an email contact for reporting issues.
* **Evidence:** Pull request: #40

#### Added missing data validation check

* **Feedback:** Project was missing a check for target–feature leakage.
* **Change made:** Added a target–feature leakage check using `pointblank`.
* **Evidence:** Pull request: #59

#### Added justification for data validation check

* **Feedback:** You should define and justify thresholds for expected distribution carefully.
* **Change made:** Added a comment that explains reasoning behind checks performed
* **Evidence:** scripts/02-validation.py line 227 - 230

#### Added methods and results header
* **Feedback:** Please also clearly add Methods section and Results section title
* **Change made:** Added methods and results header
* **Evidence:** commit 92fd48c

#### Added Creative Commons license
* **Feedback:** No Creative Commons license (for project report) was specified. LICENSE file did not contain names of all group members
* **Change made:** Added Creative Commons license
* **Evidence:** LICENSE file

#### Fixed plot axes
* **Feedback:** There are variable names displayed on the plot, please use phrases separated by whitespace, not underscore
* **Change made:** Used white space instead of underscore
* **Evidence:** scripts/04-eda.py line 19 and 56

#### Fixed font size for plots
* **Feedback:** The font size is too small for plots, especially the bar plots
* **Change made:** plot is bigger when redered quarto
* **Evidence:** reports/relationship_quality_analysis.pdf
---
