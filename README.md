# ADMET-DS-Platform
ADMET-DS-Platform
# ADMET-DS-Platform: Unified Multi-Parametric ADMET Prediction

An integrated computational platform for multi-parametric ADMET risk assessment and developability scoring using hybrid machine learning models and mechanistic expert filters.

---

## Platform Overview

- **Total Platform Outputs:** 68
  - 44 Machine Learning Models (34 Classification + 10 Continuous Regression)
  - 10 Deterministic Physicochemical Descriptors
  - 9 Rule-Based Medicinal Chemistry & Structural Alert Filters
  - 5 Composite Developability Scoring Indices (PK-DI, CC-SI, GO-SI, MC-DI, and ADMET-DS)
- **Feature Space:** 1,035 hybrid features (1,024-bit Morgan Fingerprint [radius 2] + 11 RDKit physicochemical descriptors)
- **Mechanistic Constraints:** Enforces physiological pH 7.4 ionization speciation, cellular permeability boundaries, and pharmacophore checkpoints (`expert_filters.py`).

---

## Repository Structure

- `app.py`: Production web server and composite scoring engine.
- `expert_filters.py`: Deterministic expert rules, ionization state speciation, and structural boundary overrides.
- `features.py`: Hybrid 1,035-dimensional feature generation pipeline.
- `requirements.txt`: Python package environment dependencies.
- **Releases (`v1.0.0`):** Curated training, validation, and test datasets (`Datasets.zip`, 106 MB).

---

## Installation & Local Execution

```bash
# 1. Clone repository
git clone https://github.com/biotechnology199624-png/ADMET-DS-Platform.git
cd ADMET-DS-Platform

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch web server
python app.py
## Citation

