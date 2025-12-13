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

Addressed TA and Peer comments (see GitHub Issues)
- dec966c5e47ce802f20a29ae6a02d35192a80acb

Added Dummy variable (as per Peer Christine requested)
- d46bbd5d996dab91aa3a0b3e32982f941c09ed77
