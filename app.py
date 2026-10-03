# app.py - Master Production Webserver (ML + Expert Overrides)
import os
import sys
import io
import math
import re
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

try:
    import xgboost
except ImportError:
    pass

app = Flask(__name__)

# Append central Webserver directory
WEBSERVER_PATH = r"D:\d_drive_data\Sanjay\PhD\Sanjay PhD\PhD Research Work\Objective 2 Research\Datasets\Webserver"
if WEBSERVER_PATH not in sys.path:
    sys.path.append(WEBSERVER_PATH)

try:
    from features import calculate_hybrid_features
    from expert_filters import (
        evaluate_ames_expert_rules, 
        calculate_logd_74, 
        apply_renal_transporter_filters,
        apply_hepatic_transporter_filters,
        apply_bcrp_filters,
        apply_bcrp_inhibitor_filters,
        apply_pgp_inhibitor_filters,
        apply_pampa_filters,
        apply_herg_filters,
        apply_clintox_filters,
        apply_ppb_continuous_filters,
        apply_vdss_continuous_filters,
        apply_halflife_continuous_filters,
        apply_hia_filters,
        apply_clearance_continuous_filters,
        apply_solubility_continuous_filters,
        apply_bioavailability_filters,
        apply_carcinogenicity_filters,
        apply_sensitization_filters,
        apply_mitotox_filters,
        apply_bsep_filters,
        apply_mdck_filters,
        apply_logbb_continuous_filters,
        apply_rbp_filters,
        apply_cyp_secondary_filters,
        apply_fubrain_filters,
        apply_mtd_filters
    )
except ModuleNotFoundError:
    print(f"Startup Warning: Could not locate 'features.py' or 'expert_filters.py' in {WEBSERVER_PATH}")

try:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    brenk_catalog = FilterCatalog(params)
    print("Startup: BRENK structural catalog loaded successfully.")
except Exception as e:
    print(f"Startup Warning: Failed to load BRENK catalog: {e}")
    brenk_catalog = None

# ==============================================================================
# AUTO-DISCOVERY MODEL REGISTRY (Indexes all 97 models in ADMET Property Documented)
# ==============================================================================
BASE_DOC_DIR = r"D:\d_drive_data\Sanjay\PhD\Sanjay PhD\PhD Research Work\Objective 2 Research\Datasets\ADMET Property Documented"

MODEL_REGISTRY = {}
for root, dirs, files in os.walk(BASE_DOC_DIR):
    for f in files:
        if f.endswith(".pkl") or f.endswith(".joblib"):
            MODEL_REGISTRY[f.lower()] = os.path.join(root, f)

def load_model(filename):
    path = MODEL_REGISTRY.get(filename.lower())
    if path and os.path.exists(path):
        try:
            obj = joblib.load(path)
            if isinstance(obj, dict) and 'model' in obj:
                return obj['model']
            return obj
        except Exception as e:
            print(f"Warning: Failed loading {filename}: {e}")
            return None
    print(f"Warning: Model file '{filename}' not found in registry.")
    return None

# Load Core Models
ames_model = load_model("ames_rf_model.joblib")
pka_acid_model = load_model("pka_acid_regressor.joblib")
pka_base_model = load_model("pka_base_regressor.joblib")
caco2_model = load_model("xgb_caco2_model.pkl")
pgp_model = load_model("xgb_pgp_model.pkl")
dili_model = load_model("xgb_dili_model.pkl")
ld50_model = load_model("xgb_ld50_model.pkl")
ppb_model = load_model("xgb_ppb_clf.pkl")
ppb_scaler = load_model("ppb_scaler.pkl")
bbb_model = load_model("bbb_classifier.joblib")

# Load CYP450 Models
cyp_1a2_inh = load_model("xgb_cyp1a2_inhibitor_model_hybrid.pkl")
cyp_2c9_inh = load_model("xgb_cyp2c9_inhibitor_model_hybrid.pkl")
cyp_2c19_inh = load_model("xgb_cyp2c19_inhibitor_model_hybrid.pkl")
cyp_2d6_inh = load_model("xgb_cyp2d6_inhibitor_model_hybrid.pkl")
cyp_3a4_inh = load_model("xgb_cyp3a4_inhibitor_model_hybrid.pkl")
cyp_2c8_inh = load_model("xgb_cyp2c8_inhibitor_model.pkl")
cyp_2e1_inh = load_model("xgb_cyp2e1_inhibitor_model.pkl")
cyp_2c9_sub = load_model("xgb_cyp2c9_substrate_model_hybrid.pkl")
cyp_2d6_sub = load_model("xgb_cyp2d6_substrate_model_hybrid.pkl")
cyp_3a4_sub = load_model("xgb_cyp3a4_substrate_model_hybrid.pkl")

# Load Transporters & Kinetics
pgp_inhibitor_model = load_model("xgb_pgp_inhibitor_model.pkl")
bcrp_model = load_model("xgb_bcrp_model.pkl")
bcrp_inhibitor_model = load_model("xgb_bcrp_inhibitor_model.pkl")
pampa_model = load_model("xgb_pampa_model.pkl")
oat1_model = load_model("xgb_oat1_model.pkl")
oat3_model = load_model("xgb_oat3_model.pkl")
oct1_model = load_model("xgb_oct1_model.pkl")
oct2_model = load_model("xgb_oct2_model.pkl")
mate1_model = load_model("xgb_mate1_model.pkl")
mate2k_model = load_model("xgb_mate2k_model.pkl")
oatp1b1_model = load_model("xgb_oatp1b1_model.pkl")
oatp1b3_model = load_model("xgb_oatp1b3_model.pkl")
bsep_model = load_model("xgb_bsep_model.pkl")
mdck_model = load_model("mdck_model.joblib")

# Load Safety & Toxicity Models
herg_model = load_model("xgb_herg_classifier_model.pkl") or load_model("xgb_herg_model.pkl")
clintox_model = load_model("xgb_clintox_model.pkl")
logbb_model = load_model("logbb_regressor.joblib")
rbp_model = load_model("Rbp_Hybrid_Model.pkl") or load_model("Rbp_classifier.pkl")
fubrain_model = load_model("xgb_fu_brain_model.pkl")
mtd_model = load_model("xgb_mtd_model.pkl")

# Load Continuous Regressors
fu_regressor = load_model("xgb_fu_regressor.pkl")
vdss_regressor = load_model("xgb_vdss_regressor.pkl")
halflife_regressor = load_model("xgb_halflife_regressor.pkl")
clearance_regressor = load_model("xgb_clearance_regressor.pkl")
solubility_regressor = load_model("xgb_solubility_regressor.pkl")
hia_model = load_model("xgb_hia_model.pkl")
bioavailability_model = load_model("xgb_bioavailability_model.pkl")
carcinogens_model = load_model("xgb_carcinogens_model.pkl")
sensitization_model = load_model("xgb_sensitization_model.pkl")
mitotox_model = load_model("xgb_mitotox_model.pkl")

print(f"Startup: Successfully indexed {len(MODEL_REGISTRY)} models from ADMET Property Documented.")

# ==========================================
# DRUG-LIKENESS CALCULATORS
# ==========================================
def calculate_sa_score(mol):
    if mol is None or mol.GetNumAtoms() == 0:
        return 1.0
    n_atoms = mol.GetNumAtoms()
    n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    ri = mol.GetRingInfo()
    n_rings = ri.NumRings()
    n_spiro, n_bridgehead = 0, 0
    try:
        from rdkit.Chem import rdMolDescriptors
        n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    except Exception:
        pass
    n_macrocycles = sum(1 for ring in ri.AtomRings() if len(ring) > 8)
    complexity_penalty = (n_atoms * 0.005) + (n_rings * 0.05) + (n_chiral * 0.2) + (n_spiro * 0.2) + (n_bridgehead * 0.2) + (n_macrocycles * 0.2)
    fsp3 = Descriptors.FractionCSP3(mol)
    halogens = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[Cl,Br,I,F]")))
    base_contrib = 1.8 + (1.2 * fsp3) + (halogens * 0.1)
    return round(max(1.0, min(10.0, base_contrib + complexity_penalty)), 2)

def calculate_mce18(mol):
    fsp3 = Descriptors.FractionCSP3(mol)
    chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    return round(10.0 + (35.0 * fsp3) + (12.0 * chiral), 1)

def evaluate_all_drug_likeness_rules(mol):
    mw = Descriptors.MolWt(mol)
    logP = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    mr = Descriptors.MolMR(mol)
    total_atoms = Chem.AddHs(mol).GetNumAtoms()

    lipinski_violations = []
    if mw > 500: lipinski_violations.append("MW > 500")
    if logP > 5.0: lipinski_violations.append("LogP > 5.0")
    if hbd > 5: lipinski_violations.append("HBD > 5")
    if hba > 10: lipinski_violations.append("HBA > 10")
    lipinski_str = "Yes; 0 violation" if not lipinski_violations else f"No; {len(lipinski_violations)} violations: {', '.join(lipinski_violations)}"

    ghose_violations = []
    if not (160 <= mw <= 480): ghose_violations.append("MW")
    if not (-0.4 <= logP <= 5.6): ghose_violations.append("LogP")
    if not (40 <= mr <= 130): ghose_violations.append("MR")
    if not (20 <= total_atoms <= 70): ghose_violations.append("Atoms")
    ghose_str = "Yes" if not ghose_violations else "No; 1 violation"

    veber_str = "Yes" if (rot_bonds <= 10 and tpsa <= 140) else "No; 1 violation"
    egan_str = "Yes" if (logP <= 5.88 and tpsa <= 131.6) else "No; 1 violation"
    muegge_str = "Yes" if (200 <= mw <= 600 and tpsa <= 150) else "No; 1 violation"
    lead_str = "Yes; 0 violation" if (250 <= mw <= 350 and -0.9 <= logP <= 3.5 and rot_bonds <= 7) else "No; 1 violation"

    return {
        "lipinski": lipinski_str, "ghose": ghose_str, "veber": veber_str,
        "egan": egan_str, "muegge": muegge_str, "leadlikeness": lead_str,
        "mw": mw, "logp": logP, "tpsa": tpsa, "hbd": hbd, "hba": hba,
        "rot_bonds": rot_bonds, "mr": mr, "total_atoms": total_atoms,
        "n_rings": mol.GetRingInfo().NumRings(),
        "c_atoms": len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#6]"))),
        "hetero_atoms": len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6]])
    }

def evaluate_brenk_alerts(mol):
    if brenk_catalog is None: return "0 alert"
    if brenk_catalog.HasMatch(mol):
        matches = brenk_catalog.GetMatches(mol)
        alerts = list(set([m.GetDescription() for m in matches]))
        return f"{len(alerts)} alert" + ("s" if len(alerts) != 1 else "") + f": {', '.join(alerts)}"
    return "0 alert"

# ==========================================
# PREDICTION PIPELINE (WITH EXPERT OVERRIDES)
# ==========================================
def _run_prediction(smiles, mol):
    mw = float(Descriptors.MolWt(mol))
    logP = float(Descriptors.MolLogP(mol))
    tpsa = float(Descriptors.TPSA(mol))
    fsp3 = float(Descriptors.FractionCSP3(mol))
    chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    sa_score = calculate_sa_score(mol)
    brenk_str = evaluate_brenk_alerts(mol)
    likeness = evaluate_all_drug_likeness_rules(mol)

    from rdkit.Chem import DataStructs
    fp_2048 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    features_2048 = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp_2048, features_2048)

    try:
        features = calculate_hybrid_features(smiles)
    except Exception:
        features = np.zeros((1035,))
    if features is None:
        features = np.zeros((1035,))

    results = {
        'geometry': {"mw": round(mw, 2), "fsp3": round(fsp3, 3), "chiral_centers": chiral_centers}
    }

    # 1. pKa, logD7.4, & Solubility with Expert Filters
    pred_acid = float(pka_acid_model.predict([features])[0]) if pka_acid_model else 12.00
    pred_base = float(pka_base_model.predict([features])[0]) if pka_base_model else 3.00
    logD, ion_class, note = calculate_logd_74(smiles, logP, pred_acid, pred_base)

    pred_logs = float(solubility_regressor.predict([features])[0]) if solubility_regressor else -2.649
    final_logs, logs_override_msg = apply_solubility_continuous_filters(pred_logs)

    results['pka_engine'] = {
        "predicted_acidic_pka": round(pred_acid, 2),
        "predicted_basic_pka": round(pred_base, 2),
        "calculated_logD_74": logD,
        "calculated_logP": round(logP, 2),
        "ionization_class": ion_class,
        "mechanistic_note": note,
        "predicted_logs": round(final_logs, 3),
        "predicted_sol_mg_ml": round((10**final_logs) * mw, 3),
        "solubility_class": "Highly Soluble" if final_logs >= 0.0 else \
                            "Soluble" if final_logs >= -2.0 else \
                            "Moderately Soluble" if final_logs >= -4.0 else \
                            "Poorly Soluble" if final_logs >= -6.0 else "Insoluble",
        "sol_override_status": logs_override_msg
    }

    # 2. Absorption & Efflux
    caco2_val = float(caco2_model.predict([features])[0]) if caco2_model else (0.55 if logP > 1.0 else 0.15)
    results['caco2'] = {"value": round(caco2_val, 4)}

    is_pgp_sub = bool(pgp_model.predict([features])[0] == 1) if pgp_model else bool(logP > 2.0 and mw > 350)
    results['pgp'] = {"active": is_pgp_sub}
    results['efflux_ratio'] = {"value": 2.8 if is_pgp_sub else 1.1}

    raw_pgp_inh = float(pgp_inhibitor_model.predict_proba([features])[0][1]) if pgp_inhibitor_model else 0.5
    final_pgp_inh, pgp_inh_msg = apply_pgp_inhibitor_filters(smiles, raw_pgp_inh, mw, logP)
    results['pgp_inhibitor'] = {
        "raw_probability": round(raw_pgp_inh, 3),
        "final_probability": round(final_pgp_inh, 3),
        "classification": "Inhibitor" if final_pgp_inh >= 0.5 else "Non-Inhibitor",
        "override_status": pgp_inh_msg
    }

    # Distribution & Systemic Kinetics
    ppb_prediction = "Medium Binding (50%-90%)"
    ppb_probs = [33.3, 33.3, 33.3]
    if ppb_model is not None and ppb_scaler is not None:
        phys_scaled = ppb_scaler.transform([features[:11]])[0]
        X_final = np.hstack([phys_scaled * 2.0, features[11:]])
        probs = ppb_model.predict_proba([X_final])[0]
        classes = ['Low Binding (≤50%)', 'Medium Binding (50%-90%)', 'High Binding (>90%)']
        ppb_prediction = classes[int(np.argmax(probs))]
        ppb_probs = [float(probs[0]) * 100, float(probs[1]) * 100, float(probs[2]) * 100]

    raw_fu = float(fu_regressor.predict([features])[0]) if fu_regressor else 0.5
    final_fu, fu_msg = apply_ppb_continuous_filters(smiles, raw_fu, logP, tpsa)

    raw_log_vdss = float(vdss_regressor.predict([features])[0]) if vdss_regressor else -0.022
    final_log_vdss, vdss_msg = apply_vdss_continuous_filters(raw_log_vdss)

    raw_log_halflife = float(halflife_regressor.predict([features])[0]) if halflife_regressor else 0.623
    final_log_halflife, halflife_msg = apply_halflife_continuous_filters(raw_log_halflife)

    raw_log_cl = float(clearance_regressor.predict([features])[0]) if clearance_regressor else 1.322
    final_log_cl, cl_msg = apply_clearance_continuous_filters(raw_log_cl)

    results['ppb'] = {
        "prediction": ppb_prediction,
        "low_prob": round(ppb_probs[0], 1), "med_prob": round(ppb_probs[1], 1), "high_prob": round(ppb_probs[2], 1),
        "predicted_fu": round(final_fu, 4), "predicted_ppb_percent": round((1.0 - final_fu) * 100, 1),
        "fu_override_status": fu_msg,
        "predicted_log_vdss": round(final_log_vdss, 4), "predicted_vdss_raw": round(10**final_log_vdss, 4),
        "vdss_override_status": vdss_msg,
        "predicted_log_halflife": round(final_log_halflife, 4), "predicted_halflife_raw": round(10**final_log_halflife, 4),
        "halflife_override_status": halflife_msg,
        "predicted_log_cl": round(final_log_cl, 4), "predicted_cl_raw": round(10**final_log_cl, 4),
        "cl_override_status": cl_msg
    }

    raw_bcrp = float(bcrp_model.predict_proba([features])[0][1]) if bcrp_model else 0.5
    final_bcrp, bcrp_msg = apply_bcrp_filters(smiles, raw_bcrp)
    results['bcrp'] = {
        "raw_probability": round(raw_bcrp, 3), "final_probability": round(final_bcrp, 3),
        "classification": "Substrate" if final_bcrp >= 0.5 else "Non-Substrate",
        "override_status": bcrp_msg
    }

    raw_bcrp_inh = float(bcrp_inhibitor_model.predict_proba([features])[0][1]) if bcrp_inhibitor_model else 0.5
    final_bcrp_inh, bcrp_inh_msg = apply_bcrp_inhibitor_filters(smiles, raw_bcrp_inh, mw, logP)
    results['bcrp_inhibitor'] = {
        "raw_probability": round(raw_bcrp_inh, 3), "final_probability": round(final_bcrp_inh, 3),
        "classification": "Inhibitor" if final_bcrp_inh >= 0.5 else "Non-Inhibitor",
        "override_status": bcrp_inh_msg
    }

    raw_pampa = float(pampa_model.predict_proba([features])[0][1]) if pampa_model else 0.5
    final_pampa, pampa_msg = apply_pampa_filters(smiles, raw_pampa, mw, tpsa, logP)
    results['pampa'] = {
        "raw_probability": round(raw_pampa, 3), "final_probability": round(final_pampa, 3),
        "classification": "High Permeability" if final_pampa >= 0.5 else "Low/Mod Permeability",
        "override_status": pampa_msg
    }

    raw_hia = float(hia_model.predict_proba([features])[0][1]) if hia_model else 0.5
    final_hia, hia_msg = apply_hia_filters(smiles, raw_hia)
    results['hia'] = {
        "raw_probability": round(raw_hia, 3), "final_probability": round(final_hia, 3),
        "classification": "High Absorption" if final_hia >= 0.5 else "Low Absorption",
        "override_status": hia_msg
    }

    raw_f = float(bioavailability_model.predict_proba([features])[0][1]) if bioavailability_model else 0.5
    final_f, f_msg = apply_bioavailability_filters(smiles, raw_f, final_hia, final_log_cl)
    results['bioavailability'] = {
        "raw_probability": round(raw_f, 3), "final_probability": round(final_f, 3),
        "classification": "Bioavailable (F >= 30%)" if final_f >= 0.5 else "Poorly Bioavailable (F < 30%)",
        "override_status": f_msg
    }

    raw_mdck = float(mdck_model.predict_proba([features_2048])[0][1]) if mdck_model else 0.5
    final_mdck, mdck_msg = apply_mdck_filters(smiles, raw_mdck, mw, logP)
    results['mdck'] = {
        "raw_probability": round(raw_mdck, 3), "final_probability": round(final_mdck, 3),
        "classification": "Substrate (High Efflux)" if final_mdck >= 0.5 else "Non-Substrate (Low Efflux)",
        "override_status": mdck_msg
    }

    raw_logbb = float(logbb_model.predict([features_2048])[0]) if logbb_model else -0.078
    final_logbb, logbb_msg = apply_logbb_continuous_filters(smiles, raw_logbb, mw, tpsa, logP)
    results['logbb'] = {
        "raw_value": round(raw_logbb, 4), "final_value": round(final_logbb, 4),
        "classification": "CNS Penetrant (++)" if final_logbb >= -1.00 else "Non-Penetrant (--)",
        "override_status": logbb_msg
    }

    raw_rbp = 0.15 if logP < 3.0 else 0.55
    final_rbp, rbp_msg = apply_rbp_filters(smiles, raw_rbp, logP)
    results['rbp'] = {
        "raw_probability": round(raw_rbp, 3), "final_probability": round(final_rbp, 3),
        "classification": "High Sequestration Risk (Fail)" if final_rbp >= 0.50 else "Low Sequestration Risk (Pass)",
        "override_status": rbp_msg
    }

    raw_log_fubrain = float(fubrain_model.predict([features_2048])[0]) if fubrain_model else -1.0
    raw_fubrain_val = max(0.001, min(100.0, float((10**raw_log_fubrain) * 100.0)))
    final_fubrain, fubrain_msg = apply_fubrain_filters(raw_fubrain_val, logP, smiles)
    results['fubrain'] = {
        "raw_value": round(raw_log_fubrain, 4), "final_value": round(final_fubrain, 3),
        "override_status": fubrain_msg
    }

    # Transporters with expert ionization boundary filters
    raw_oat1 = float(oat1_model.predict_proba([features])[0][1]) if oat1_model else 0.5
    raw_oat3 = float(oat3_model.predict_proba([features])[0][1]) if oat3_model else 0.5
    raw_oct2 = float(oct2_model.predict_proba([features_2048])[0][1]) if oct2_model else 0.5
    raw_mate1 = float(mate1_model.predict_proba([features_2048])[0][1]) if mate1_model else 0.5
    raw_mate2k = float(mate2k_model.predict_proba([features_2048])[0][1]) if mate2k_model else 0.5

    f_oat1, f_oat3, f_oct2, f_mate1, f_mate2k, renal_msg = apply_renal_transporter_filters(
        smiles=smiles, pka_acid=pred_acid, pka_base=pred_base,
        raw_oat1_prob=raw_oat1, raw_oat3_prob=raw_oat3, raw_oct2_prob=raw_oct2,
        raw_mate1_prob=raw_mate1, raw_mate2k_prob=raw_mate2k
    )

    results['renal_transporters'] = {
        "oat1": {"raw_probability": round(raw_oat1, 3), "final_probability": round(f_oat1, 3), "classification": "Substrate" if f_oat1 >= 0.5 else "Non-Substrate"},
        "oat3": {"raw_probability": round(raw_oat3, 3), "final_probability": round(f_oat3, 3), "classification": "Substrate" if f_oat3 >= 0.5 else "Non-Substrate"},
        "oct2": {"raw_probability": round(raw_oct2, 3), "final_probability": round(f_oct2, 3), "classification": "Substrate" if f_oct2 >= 0.5 else "Non-Substrate"},
        "mate1": {"raw_probability": round(raw_mate1, 3), "final_probability": round(f_mate1, 3), "classification": "Inhibitor" if f_mate1 >= 0.5 else "Non-Inhibitor"},
        "mate2k": {"raw_probability": round(raw_mate2k, 3), "final_probability": round(f_mate2k, 3), "classification": "Inhibitor" if f_mate2k >= 0.5 else "Non-Inhibitor"},
        "override_status": renal_msg
    }

    raw_oatp1b1 = float(oatp1b1_model.predict_proba([features_2048])[0][1]) if oatp1b1_model else 0.5
    raw_oatp1b3 = float(oatp1b3_model.predict_proba([features_2048])[0][1]) if oatp1b3_model else 0.5
    raw_oct1 = float(oct1_model.predict_proba([features_2048])[0][1]) if oct1_model else 0.5
    raw_bsep = float(bsep_model.predict_proba([features])[0][1]) if bsep_model else 0.5

    f_oatp1b1, f_oatp1b3, f_oct1, hepatic_msg = apply_hepatic_transporter_filters(
        smiles=smiles, pka_acid=pred_acid, pka_base=pred_base,
        raw_oatp1b1_prob=raw_oatp1b1, raw_oatp1b3_prob=raw_oatp1b3, raw_oct1_prob=raw_oct1
    )
    final_bsep, bsep_msg = apply_bsep_filters(smiles=smiles, pka_acid=pred_acid, pka_base=pred_base, raw_bsep_prob=raw_bsep)

    results['hepatic_transporters'] = {
        "oatp1b1": {"raw_probability": round(raw_oatp1b1, 3), "final_probability": round(f_oatp1b1, 3), "classification": "Inhibitor" if f_oatp1b1 >= 0.5 else "Non-Inhibitor"},
        "oatp1b3": {"raw_probability": round(raw_oatp1b3, 3), "final_probability": round(f_oatp1b3, 3), "classification": "Inhibitor" if f_oatp1b3 >= 0.5 else "Non-Inhibitor"},
        "oct1": {"raw_probability": round(raw_oct1, 3), "final_probability": round(f_oct1, 3), "classification": "Substrate" if f_oct1 >= 0.5 else "Non-Substrate"},
        "bsep": {"raw_probability": round(raw_bsep, 3), "final_probability": round(final_bsep, 3), "classification": "Inhibitor" if final_bsep >= 0.5 else "Non-Inhibitor"},
        "override_status": hepatic_msg, "bsep_override_status": bsep_msg
    }

    # CYP450 Matrix
    raw_2c8 = float(cyp_2c8_inh.predict_proba([features_2048])[0][1]) if cyp_2c8_inh else 0.15
    raw_2e1 = float(cyp_2e1_inh.predict_proba([features_2048])[0][1]) if cyp_2e1_inh else 0.15
    f_2c8, f_2e1, cyp_sec_msg = apply_cyp_secondary_filters(smiles, raw_2c8, raw_2e1, mw, tpsa)

    results['cyp450'] = {
        "inhibitor_1a2": bool(cyp_1a2_inh.predict([features])[0] == 1) if cyp_1a2_inh else False,
        "inhibitor_2c9": bool(cyp_2c9_inh.predict([features])[0] == 1) if cyp_2c9_inh else False,
        "inhibitor_2c19": bool(cyp_2c19_inh.predict([features])[0] == 1) if cyp_2c19_inh else False,
        "inhibitor_2d6": bool(cyp_2d6_inh.predict([features])[0] == 1) if cyp_2d6_inh else False,
        "inhibitor_3a4": bool(cyp_3a4_inh.predict([features])[0] == 1) if cyp_3a4_inh else False,
        "inhibitor_2c8": bool(f_2c8 >= 0.50), "inhibitor_2e1": bool(f_2e1 >= 0.50),
        "inhibitor_2c8_prob": round(f_2c8, 3), "inhibitor_2e1_prob": round(f_2e1, 3),
        "override_status": cyp_sec_msg,
        "substrate_2c9": bool(cyp_2c9_sub.predict([features])[0] == 1) if cyp_2c9_sub else False,
        "substrate_2d6": bool(cyp_2d6_sub.predict([features])[0] == 1) if cyp_2d6_sub else False,
        "substrate_3a4": bool(cyp_3a4_sub.predict([features])[0] == 1) if cyp_3a4_sub else False
    }

    # Toxicology & Safety Overrides
    raw_ames_prob = float(ames_model.predict_proba([features])[0][1]) if ames_model else 0.12
    has_alert, ames_msg = evaluate_ames_expert_rules(smiles)
    is_mutagenic = (raw_ames_prob >= 0.5) or has_alert
    results['ames'] = {
        "probability": round(raw_ames_prob, 3), "prediction": "Mutagenic (+)" if is_mutagenic else "Non-mutagenic (-)",
        "expert_alert": ames_msg, "status": "red" if is_mutagenic else "green"
    }

    raw_herg = bool(herg_model.predict([features_2048])[0] == 1) if herg_model else False
    is_herg_toxic, herg_msg = apply_herg_filters(smiles, raw_herg)
    results['herg'] = {"cardiotox": is_herg_toxic, "override_status": herg_msg}

    raw_carc = float(carcinogens_model.predict_proba([features])[0][1]) if carcinogens_model else 0.5
    final_carc, carc_msg = apply_carcinogenicity_filters(smiles, raw_carc)
    results['carcinogens'] = {"raw_probability": round(raw_carc, 3), "final_probability": round(final_carc, 3), "classification": "Carcinogenic (+)" if final_carc >= 0.5 else "Non-carcinogenic (-)", "override_status": carc_msg}

    raw_sens = float(sensitization_model.predict_proba([features])[0][1]) if sensitization_model else 0.5
    final_sens, sens_msg = apply_sensitization_filters(smiles, raw_sens, final_logs)
    results['sensitization'] = {"raw_probability": round(raw_sens, 3), "final_probability": round(final_sens, 3), "classification": "Sensitizer" if final_sens >= 0.5 else "Non-Sensitizer", "override_status": sens_msg}

    raw_mito = float(mitotox_model.predict_proba([features])[0][1]) if mitotox_model else 0.5
    final_mito, mito_msg = apply_mitotox_filters(smiles, raw_mito, final_logs, logP)
    results['mitotox'] = {"raw_probability": round(raw_mito, 3), "final_probability": round(final_mito, 3), "classification": "Toxic (+)" if final_mito >= 0.5 else "Safe (-)", "override_status": mito_msg}

    is_dili = bool(dili_model.predict([features])[0] == 1) if dili_model else False
    if is_dili and (logP < 3.0 and mw < 250):
        is_dili = False
    results['dili'] = {"hepatotox": is_dili}

    pld50 = float(ld50_model.predict([features])[0]) if ld50_model else 2.5
    ld50_mg = (10 ** (3.0 - pld50)) * mw
    ghs = 1 if ld50_mg <= 5 else 2 if ld50_mg <= 50 else 3 if ld50_mg <= 300 else 4 if ld50_mg <= 2000 else 5
    results['ld50'] = {"value": round(float(ld50_mg), 1), "ghs_category": ghs}

    raw_clintox = float(clintox_model.predict_proba([features])[0][1]) if clintox_model else 0.5
    final_clintox, clintox_msg = apply_clintox_filters(smiles, raw_clintox, is_mutagenic, is_herg_toxic)
    results['clintox'] = {"raw_probability": round(raw_clintox, 3), "final_probability": round(final_clintox, 3), "classification": "High Attrition Risk (Fail)" if final_clintox >= 0.5 else "Low Attrition Risk (Pass)", "override_status": clintox_msg}

    raw_mtd = float(mtd_model.predict_proba([features_2048])[0][1]) if mtd_model else 0.5
    final_mtd, mtd_msg = apply_mtd_filters(smiles, raw_mtd, logP, mw)
    results['mtd'] = {"raw_probability": round(raw_mtd, 3), "final_probability": round(final_mtd, 3), "classification": "High Toxicity (Low MTD)" if final_mtd >= 0.5 else "Low Toxicity (High MTD)", "override_status": mtd_msg}

    # MedChem & Developability Score
    results['med_chem'] = {
        "qed": float(QED.qed(mol)), "mce_18": calculate_mce18(mol), "sa_score": sa_score, "brenk": brenk_str,
        "lipinski": likeness["lipinski"], "ghose": likeness["ghose"], "veber": likeness["veber"],
        "egan": likeness["egan"], "muegge": likeness["muegge"], "leadlikeness": likeness["leadlikeness"],
        "mr": round(likeness["mr"], 2), "total_atoms": likeness["total_atoms"], "hetero_atoms": likeness["hetero_atoms"],
        "c_atoms": likeness["c_atoms"], "rot_bonds": likeness["rot_bonds"], "tpsa": round(likeness["tpsa"], 2),
        "logp": round(likeness["logp"], 2), "mw": round(likeness["mw"], 2), "pains_alerts": 0
    }

    pk_di = 1.0 + (2.5 if final_logs >= -4.0 else 1.5 if final_logs >= -6.0 else 0.5) + (2.5 if final_hia >= 0.5 else 0.5) + (2.5 if final_f >= 0.5 else 0.5) + (1.5 if results['ppb']['predicted_cl_raw'] <= 20.0 else 0.25)
    pk_di = max(1.0, min(10.0, pk_di))

    cc_si = 0.1 + (3.3 if not is_herg_toxic else 0.1) + (3.3 if final_mtd < 0.5 else 0.5) + (3.3 if final_clintox < 0.5 else 0.5)
    cc_si = max(1.0, min(10.0, cc_si))

    go_si = 0.1 + (3.3 if not is_mutagenic else 0.1) + (3.3 if not is_dili else 0.5) + (3.3 if final_mito < 0.5 else 0.5)
    go_si = max(1.0, min(10.0, go_si))

    mc_di = 1.0 + (results['med_chem']['qed'] * 3.0) + (3.0 if sa_score < 3.5 else 1.5 if sa_score < 5.0 else 0.5) + (3.0 if brenk_str.startswith("0 alert") else 1.0)
    mc_di = max(1.0, min(10.0, mc_di))

    admet_ds = float((pk_di * cc_si * go_si * mc_di) ** 0.25)

    results['expert_score'] = {
        "admet_ds": round(admet_ds, 2), "pk_di": round(pk_di, 2), "cc_si": round(cc_si, 2),
        "go_si": round(go_si, 2), "mc_di": round(mc_di, 2),
        "interpretation": "Excellent (ADMET-DS >= 8.0)" if admet_ds >= 8.0 else "Good (ADMET-DS >= 6.0)" if admet_ds >= 6.0 else "Moderate (ADMET-DS >= 4.0)" if admet_ds >= 4.0 else "Poor/High Risk"
    }

    return results

# ==========================================
# FLASK ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    smiles = request.form.get('smiles', '').strip()
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None: return jsonify({"error": "Invalid SMILES structure."}), 400
    return jsonify(_run_prediction(smiles, mol))

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    smiles_text = request.form.get('smiles_list', '').strip()
    smiles_list = [s.strip() for s in re.split(r'[\n\r,;\t]+', smiles_text) if s.strip()]
    seen = set()
    smiles_list = [s for s in smiles_list if not (s in seen or seen.add(s))][:25]

    results_list = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            results_list.append({"smiles": s, "error": "Invalid chemical structure"})
            continue
        try:
            res = _run_prediction(s, mol)
            res["smiles"] = s
            results_list.append(res)
        except Exception as e:
            results_list.append({"smiles": s, "error": str(e)})

    return jsonify({"results": results_list})

@app.route('/export_excel', methods=['POST'])
def export_excel():
    data = request.json or {}
    results_list = data.get("results", [])
    flat_results = []
    for item in results_list:
        if "error" in item: continue
        flat_results.append({
            "SMILES": item.get("smiles", ""),
            "Molecular Weight (g/mol)": item.get("geometry", {}).get("mw", ""),
            "LogP": item.get("pka_engine", {}).get("calculated_logP", ""),
            "logD7.4": item.get("pka_engine", {}).get("calculated_logD_74", ""),
            "Solubility (logS)": item.get("pka_engine", {}).get("predicted_logs", ""),
            "Caco-2 Permeability": item.get("caco2", {}).get("value", ""),
            "P-gp Substrate": "Substrate" if item.get("pgp", {}).get("active") else "Non-Substrate",
            "P-gp Inhibitor": item.get("pgp_inhibitor", {}).get("classification", ""),
            "BCRP Substrate": item.get("bcrp", {}).get("classification", ""),
            "PAMPA": item.get("pampa", {}).get("classification", ""),
            "HIA": item.get("hia", {}).get("classification", ""),
            "Bioavailability": item.get("bioavailability", {}).get("classification", ""),
            "OAT1": item.get("renal_transporters", {}).get("oat1", {}).get("classification", ""),
            "OCT2": item.get("renal_transporters", {}).get("oct2", {}).get("classification", ""),
            "MATE1": item.get("renal_transporters", {}).get("mate1", {}).get("classification", ""),
            "MATE2K": item.get("renal_transporters", {}).get("mate2k", {}).get("classification", ""),
            "OATP1B1": item.get("hepatic_transporters", {}).get("oatp1b1", {}).get("classification", ""),
            "OATP1B3": item.get("hepatic_transporters", {}).get("oatp1b3", {}).get("classification", ""),
            "OCT1": item.get("hepatic_transporters", {}).get("oct1", {}).get("classification", ""),
            "hERG Cardiotoxicity": "High Risk" if item.get("herg", {}).get("cardiotox") else "Low Risk",
            "DILI Hepatotoxicity": "Toxic" if item.get("dili", {}).get("hepatotox") else "Safe",
            "ClinTox": item.get("clintox", {}).get("classification", ""),
            "AMES": item.get("ames", {}).get("prediction", ""),
            "Human MTD": item.get("mtd", {}).get("classification", ""),
            "Expert ADMET-DS": item.get("expert_score", {}).get("admet_ds", ""),
            "Mode": "With Expert Overrides"
        })
    df = pd.DataFrame(flat_results)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='With_Expert_Filters')
    output.seek(0)
    return send_file(output, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", as_attachment=True, download_name="admet_batch_results_with_expertfilter.xlsx")

if __name__ == "__main__":
    print("Starting Master ADMET Server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=True)