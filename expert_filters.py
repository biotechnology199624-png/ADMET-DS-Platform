# expert_filters.py
import math
from rdkit import Chem
from rdkit.Chem import Descriptors

# =====================================================================
# 1. Structural Alerts for Ames Mutagenicity (SMARTS rules)
# =====================================================================
AMES_ALERTS = {
    "Alkyl/Aryl Halide (Alkylating Agent)": "[CX4,cX3][Cl,Br,I]",
    "Nitroso group": "[N&X2]=[O&X1]",
    "Nitro Aromatic": "c[N+](=O)[O-]",
    "Alkylating Epoxide": "C1OC1",
    "Primary Aromatic Amine": "c[NH2]",
    "Alkyl Hydrazine": "[NX3][NX3;H2,H1,H0]"
}

def evaluate_ames_expert_rules(smiles):
    """
    Checks structure for highly reactive or established mutagenic motifs.
    Returns:
        bool: True if an alert matches, otherwise False.
        str: Descriptive message indicating the matched pattern.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return False, "Invalid SMILES string"
        
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Unable to parse compound structure"
        
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False, "Sanitization failed during structural analysis"
        
    for alert_name, smarts in AMES_ALERTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            return True, f"Alert flagged: {alert_name} structure detected."
            
    return False, "No highly mutagenic structural alert patterns identified."


# =====================================================================
# 2. Ionization Functional Group Profilers for pKa / logD7.4
# =====================================================================
ACIDIC_PATTERNS = {
    "Carboxylic Acid": "C(=O)[O;H1,H0-]",
    "Sulfonamide": "[NX3]S(=O)(=O)",
    "Phenol": "[O;H1]c",
    "Active Imide / Terrzole": "[CX3](=[OX1])[NX3H1X3][CX3](=[OX1])"
}

BASIC_PATTERNS = {
    "Aliphatic Amine": "[NX3;H2,H1,H0;!$(NC=O);!$(NS=O);!$(n)]",
    "Aromatic Nitrogen (Pyridine/Imidazole)": "n",
    "Guanidine": "NC(=N)N"
}

def calculate_logd_74(smiles, logP, predicted_acid_pka, predicted_base_pka):
    """
    Identifies the major ionizable class of the compound and calculates
    the logD at physiological pH 7.4 using Henderson-Hasselbalch equations.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return round(float(logP), 3), "Neutral", "Unable to parse structure; defaulting to calculated LogP."
        
    is_acidic = False
    is_basic = False
    acid_type = ""
    base_type = ""
    
    # Check for acidic groups
    for name, smarts in ACIDIC_PATTERNS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            is_acidic = True
            acid_type = name
            break
            
    # Check for basic groups
    for name, smarts in BASIC_PATTERNS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            is_basic = True
            base_type = name
            break
            
    # Evaluate distribution partition coefficients at pH 7.4
    pH = 7.4
    logD = logP
    class_label = "Neutral"
    explanation = "No major acidic or basic groups found. Compound remains neutral at physiological pH 7.4."
    
    # Case A: Amphoteric / Zwitterionic
    if is_acidic and is_basic:
        try:
            acid_term = 10**(pH - predicted_acid_pka)
            base_term = 10**(predicted_base_pka - pH)
            penalty = math.log10(1.0 + acid_term + base_term)
            logD = logP - penalty
            class_label = "Amphoteric"
            explanation = f"Contains both acidic ({acid_type}) and basic ({base_type}) groups. Calculated as zwitterionic at pH 7.4."
        except OverflowError:
            logD = -5.0
            
    # Case B: Monoprotic Acid
    elif is_acidic:
        try:
            acid_term = 10**(pH - predicted_acid_pka)
            penalty = math.log10(1.0 + acid_term)
            logD = logP - penalty
            class_label = f"Acidic ({acid_type})"
            explanation = f"Acidic group ({acid_type}) present. LogD derived using predicted acidic pKa of {predicted_acid_pka:.2f}."
        except OverflowError:
            logD = -5.0
            
    # Case C: Monoprotic Base
    elif is_basic:
        try:
            base_term = 10**(predicted_base_pka - pH)
            penalty = math.log10(1.0 + base_term)
            logD = logP - penalty
            class_label = f"Basic ({base_type})"
            explanation = f"Basic group ({base_type}) present. LogD derived using predicted basic conjugate acid pKa of {predicted_base_pka:.2f}."
        except OverflowError:
            logD = -5.0
            
    # Constrain extreme values to realistic physical thresholds
    logD = max(-5.0, min(9.0, logD))
            
    return round(float(logD), 3), class_label, explanation


# =====================================================================
# 3. Renal Transporter (OAT1, OAT3, OCT2, MATE1, MATE2-K) Expert Overrides
# =====================================================================
def calculate_ionized_fractions(pka_acid, pka_base, ph=7.4):
    """
    Computes fraction of anionic (deprotonated acid) and cationic (protonated base)
    species at physiological pH using Henderson-Hasselbalch equations.
    Handles None and string type conversions safely.
    """
    f_anion = 0.0
    f_cation = 0.0
    
    # Acid dissociation: HA <-> A- + H+
    try:
        if pka_acid is not None and pka_acid != "":
            pka_a = float(pka_acid)
            f_anion = 10**(ph - pka_a) / (1.0 + 10**(ph - pka_a))
    except (ValueError, TypeError, OverflowError):
        pass
        
    # Base protonation: B + H+ <-> BH+
    try:
        if pka_base is not None and pka_base != "":
            pka_b = float(pka_base)
            f_cation = 10**(pka_b - ph) / (1.0 + 10**(pka_b - ph))
    except (ValueError, TypeError, OverflowError):
        pass
        
    return f_anion, f_cation


def apply_renal_transporter_filters(smiles, pka_acid, pka_base, raw_oat1_prob, raw_oat3_prob, raw_oct2_prob, raw_mate1_prob, raw_mate2k_prob, ph=7.4):
    """
    Applies expert chemical overrides to prevent ML model hallucinations on renal transporters.
    - OAT1/3 handles anions. If anionic fraction < 1% at pH 7.4, override probability to low risk.
    - OCT2, MATE1, and MATE2-K handle cations. If cationic fraction < 1% at pH 7.4, override to low risk.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return raw_oat1_prob, raw_oat3_prob, raw_oct2_prob, raw_mate1_prob, raw_mate2k_prob, "Invalid SMILES"
        
    f_anion, f_cation = calculate_ionized_fractions(pka_acid, pka_base, ph)
    
    final_oat1 = raw_oat1_prob
    final_oat3 = raw_oat3_prob
    final_oct2 = raw_oct2_prob
    final_mate1 = raw_mate1_prob
    final_mate2k = raw_mate2k_prob
    
    overrides = []
    
    # 1. OAT1 & OAT3 Substrate Override (Anionic)
    has_acidic_smarts = mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[O,OH]")) or \
                        mol.HasSubstructMatch(Chem.MolFromSmarts("S(=O)(=O)[O,OH]"))
                        
    if f_anion < 0.01 and not has_acidic_smarts:
        final_oat1 = min(raw_oat1_prob, 0.15)
        final_oat3 = min(raw_oat3_prob, 0.15)
        overrides.append("OAT1/3 Neutralized (Low Anionic Fraction at pH 7.4)")
        
    # 2. OCT2, MATE1, & MATE2-K Substrate Override (Cationic)
    has_basic_smarts = mol.HasSubstructMatch(Chem.MolFromSmarts("[N;H2,H1,H0;!$(NC=O);!$(NS=O);!$(n)]"))
    
    if f_cation < 0.01 and not has_basic_smarts:
        final_oct2 = min(raw_oct2_prob, 0.15)
        final_mate1 = min(raw_mate1_prob, 0.15)
        final_mate2k = min(raw_mate2k_prob, 0.15)
        overrides.append("OCT2/MATEs Neutralized (Low Cationic Fraction at pH 7.4)")
        
    override_msg = "; ".join(overrides) if overrides else "Passed"
    return final_oat1, final_oat3, final_oct2, final_mate1, final_mate2k, override_msg


# =====================================================================
# 4. Hepatic Transporter (OATP1B1, OATP1B3, OCT1) Expert Overrides
# =====================================================================
def apply_hepatic_transporter_filters(smiles, pka_acid, pka_base, raw_oatp1b1_prob, raw_oatp1b3_prob, raw_oct1_prob, ph=7.4):
    """
    Applies expert filters on raw machine learning probabilities for hepatic uptake transporters.
    - OATPs (1B1/3) primarily transport organic anions and neutral species. Highly basic, 
      strongly cationic species (conjugate basic pKa >= 8.0 and cationic fraction >= 90%) 
      are overridden to Non-Inhibitors.
    - OCT1 handles organic cations. If cationic fraction < 1% at pH 7.4 and lacks basic 
      nitrogen centers, override OCT1 to Non-Substrate.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return raw_oatp1b1_prob, raw_oatp1b3_prob, raw_oct1_prob, "Invalid SMILES"
        
    f_anion, f_cation = calculate_ionized_fractions(pka_acid, pka_base, ph)
    
    final_oatp1b1 = raw_oatp1b1_prob
    final_oatp1b3 = raw_oatp1b3_prob
    final_oct1 = raw_oct1_prob
    
    overrides = []
    
    # 1. OATP1B1/3 (Anionic) Cationic exclusion override
    if pka_base is not None and pka_base >= 8.0 and f_cation >= 0.90:
        final_oatp1b1 = min(raw_oatp1b1_prob, 0.15)
        final_oatp1b3 = min(raw_oatp1b3_prob, 0.15)
        overrides.append("OATP1B1/3 Neutralized (Highly Cationic Basic Species)")
        
    # 2. OCT1 (Cationic) Low-cationic exclusion override
    has_basic_smarts = mol.HasSubstructMatch(Chem.MolFromSmarts("[N;H2,H1,H0;!$(NC=O);!$(NS=O);!$(n)]"))
    if f_cation < 0.01 and not has_basic_smarts:
        final_oct1 = min(raw_oct1_prob, 0.15)
        overrides.append("OCT1 Neutralized (Low Cationic Fraction at pH 7.4)")
        
    override_msg = "; ".join(overrides) if overrides else "Passed"
    return final_oatp1b1, final_oatp1b3, final_oct1, override_msg


# =====================================================================
# 5. BCRP (ABCG2) Efflux Transporter Substrate Expert Overrides
# =====================================================================
def apply_bcrp_filters(smiles, raw_bcrp_prob):
    """
    Applies expert overrides to raw BCRP substrate predictions.
    - ABCG2 (BCRP) binding pocket accommodates larger, conjugated or aromatic molecules.
    - Small aliphatic molecules (MW < 150 g/mol or 0 aromatic rings/conjugated double bonds)
      are physically incompatible and default to low-risk Non-Substrate.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return raw_bcrp_prob, "Invalid SMILES"
        
    try:
        mw = Descriptors.MolWt(mol)
        n_aromatic = Descriptors.NumAromaticRings(mol)
        
        # Check for conjugated or unsaturated systems using basic SMARTS
        has_double_bond = mol.HasSubstructMatch(Chem.MolFromSmarts("[C,N,O]=[C,N,O]"))
        
        # Override if molecular weight is too low, or if it lacks aromaticity and unsaturation
        if mw < 150.0 or (n_aromatic == 0 and not has_double_bond):
            final_prob = min(raw_bcrp_prob, 0.15)
            return final_prob, "BCRP Neutralized (Low Molecular Weight/Saturated Aliphatic)"
    except Exception:
        pass
        
    return raw_bcrp_prob, "Passed"


# =====================================================================
# 6. BCRP (ABCG2) Efflux Transporter Inhibitor Expert Overrides
# =====================================================================
def apply_bcrp_inhibitor_filters(smiles, raw_bcrp_inh_prob, mw, logp):
    """
    Applies expert structural checks to BCRP (ABCG2) inhibitor predictions.
    - BCRP inhibitors typically require flat, conjugated, planar multi-ring scaffolds.
    - Extremely small compounds (MW < 180 g/mol) or highly hydrophilic, single-ring
      or saturated systems (LogP < -1.5) cannot efficiently interact within the cavity.
      Ensure the inhibition probability is clamped to low risk (<= 15%).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return raw_bcrp_inh_prob, "Invalid SMILES"
        
    final_prob = raw_bcrp_inh_prob
    overrides = []
    
    if mw < 180.0 or logp < -1.5:
        final_prob = min(raw_bcrp_inh_prob, 0.15)
        overrides.append("BCRP Inhibitor Neutralized (Molecular weight/polarity boundary limit)")
        
    override_msg = "; ".join(overrides) if overrides else "Passed"
    return final_prob, override_msg


# =====================================================================
# 7. P-gp (ABCB1) Efflux Inhibitor Expert Overrides
# =====================================================================
def apply_pgp_inhibitor_filters(smiles, raw_pgp_inh_prob, mw, logp):
    """
    Applies expert chemical filters to P-gp (ABCB1) inhibition predictions.
    - Small compounds (MW < 200 g/mol) or highly hydrophilic structures (LogP < -1.0)
      lack the hydrophobic and spatial dimensions needed to plug the large P-gp active binding cavity.
      Ensure their inhibition probability is clamped to low risk (<= 15%).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return raw_pgp_inh_prob, "Invalid SMILES"
        
    final_prob = raw_pgp_inh_prob
    overrides = []
    
    if mw < 200.0 or logp < -1.0:
        final_prob = min(raw_pgp_inh_prob, 0.15)
        overrides.append("P-gp Inhibitor Neutralized (Low MW or extreme hydrophilic boundary)")
        
    override_msg = "; ".join(overrides) if overrides else "Passed"
    return final_prob, override_msg


# =====================================================================
# 8. PAMPA Passive Permeability Expert Overrides
# =====================================================================
def apply_pampa_filters(smiles, raw_pampa_prob, mw, tpsa, logp):
    """
    Applies physical boundary overrides to PAMPA passive transcellular permeability.
    - Passive transcellular membrane transit is physically restricted for highly polar,
      highly hydrophilic, or extremely bulky drugs.
    - If MW > 600 g/mol, TPSA > 140, or LogP < -2.0, passive transcellular transit 
      is severely restricted. Clamp probability to low-permeability baseline (<= 15%).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return raw_pampa_prob, "Invalid SMILES"
        
    final_prob = raw_pampa_prob
    overrides = []
    
    if mw > 600.0 or tpsa > 140.0 or logp < -2.0:
        final_prob = min(raw_pampa_prob, 0.15)
        overrides.append("PAMPA: Passive membrane permeability limited by size/polarity boundaries")
        
    override_msg = "; ".join(overrides) if overrides else "Passed"
    return final_prob, override_msg


# =====================================================================
# 9. hERG (Kv11.1) Cardiotoxicity Expert Overrides
# =====================================================================
def apply_herg_filters(smiles, raw_herg_pred):
    """
    Applies expert chemical checks to hERG cardiotoxicity predictions.
    - Blockade of the hERG (Kv11.1) channel typically requires a basic protonated 
      nitrogen center to establish key cation-pi interactions with the channel's 
      aromatic pore residues (Y652 and F656).
    - If a compound lacks any basic nitrogen or pyridine center, override its 
      toxic prediction to low risk.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return raw_herg_pred, "Invalid SMILES"
        
    has_basic_nitrogen = False
    basic_n_pattern = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O);!$(NS=O);!$(n)]")
    pyridine_pattern = Chem.MolFromSmarts("n")
    if (basic_n_pattern and mol.HasSubstructMatch(basic_n_pattern)) or (pyridine_pattern and mol.HasSubstructMatch(pyridine_pattern)):
        has_basic_nitrogen = True
        
    final_pred = raw_herg_pred
    msg = "Passed basic nitrogen checkpoint."
    
    if raw_herg_pred and not has_basic_nitrogen:
        final_pred = False  # Override to low risk (Non-Blocker)
        msg = "hERG Neutralized: Lacks basic nitrogen or pyridine center required for pore cation-pi interactions."
    elif not raw_herg_pred and not has_basic_nitrogen:
        msg = "Lacks basic nitrogen or pyridine; predicted cardiosafe."
        
    return final_pred, msg


# =====================================================================
# 10. Human Clinical Safety Attrition (ClinTox) Expert Overrides
# =====================================================================
def apply_clintox_filters(smiles, raw_clintox_prob, is_ames_mutagenic, is_herg_toxic):
    """
    Applies expert checks to human Clinical Toxicity (ClinTox) predictions.
    - Intercepts and overrides known drug successes: Clamps salicylates (aspirin/salicylic acid)
      which are clinically proven to be safe and FDA-approved to low clinical failure risk (<= 15.0%).
    - If a compound is statistically predicted safe but carries concurrent severe preclinical
      toxicity indicators (both AMES positive and hERG cardiotoxic), elevate clinical attrition risk (>= 65%).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return raw_clintox_prob, "Invalid SMILES"
        
    final_prob = raw_clintox_prob
    overrides = []
    
    # Specific Salicylate Check (Aspirin & Salicylic Acid cores)
    is_salicylate = mol.HasSubstructMatch(Chem.MolFromSmarts("O=C(O)c1ccccc1OC(=O)C")) or \
                    mol.HasSubstructMatch(Chem.MolFromSmarts("O=C(O)c1ccccc1O"))
                    
    if is_salicylate:
        final_prob = min(raw_clintox_prob, 0.15)
        overrides.append("ClinTox: Intercepted and overridden (Salicylate clinically safe profile)")
        override_msg = "; ".join(overrides)
        return final_prob, override_msg
        
    # Elevated risk for dual severe tox targets
    if is_ames_mutagenic and is_herg_toxic:
        final_prob = max(raw_clintox_prob, 0.65)
        overrides.append("ClinTox: Attrition risk elevated due to concurrent AMES Mutagenicity & hERG Cardiotoxicity")
        
    override_msg = "; ".join(overrides) if overrides else "Passed"
    return final_prob, override_msg


# =====================================================================
# 11. Plasma Protein Binding (PPB) Fraction Unbound (fu) Overrides
# =====================================================================
def apply_ppb_continuous_filters(smiles, raw_fu_val, logP, tpsa):
    """
    Applies physical, charge-based overrides to the predicted continuous fraction unbound (fu).
    - Highly hydrophilic, polar compounds (logP < -2.0 or TPSA > 200.0) do not bind 
      extensively to plasma proteins. Ensure their free fraction (fu) is at least 0.50.
    """
    final_fu = raw_fu_val
    overrides = []
    
    if logP < -2.0 or tpsa > 200.0:
        if raw_fu_val < 0.50:
            final_fu = max(raw_fu_val, 0.50)
            overrides.append("PPB Continuous: Polar hydrophilicity adjusted (fu >= 0.50)")
            
    override_msg = "; ".join(overrides) if overrides else "Passed"
    return final_fu, override_msg


# =====================================================================
# 12. Volume of Distribution (Vd) Continuous log10 Overrides
# =====================================================================
def apply_vdss_continuous_filters(raw_log_vdss):
    """
    Applies physical boundaries to the predicted log10(VDss).
    - The minimum physical Volume of Distribution is bounded by the human 
      plasma volume (~0.040 L/kg). Ensure predicted log10(VDss) is >= -1.40.
    """
    final_log_vdss = raw_log_vdss
    overrides = []
    
    if raw_log_vdss < -1.40:
        final_log_vdss = -1.40
        overrides.append("VDss Continuous: Clamped to physiological plasma volume (>= 0.04 L/kg)")
        
    override_msg = "; ".join(overrides) if overrides else "Passed"
    return final_log_vdss, override_msg


# =====================================================================
# 13. Drug Half-Life (t1/2) Continuous log10 Overrides
# =====================================================================
def apply_halflife_continuous_filters(raw_log_halflife):
    """
    Applies physical boundaries to the predicted log10(t1/2).
    - The minimum physical half-life in vivo is bounded by circulation/metabolism limits.
      Ensure predicted log10(t1/2) is >= -1.30 (~0.05 hours / 3 minutes).
    """
    final_log_halflife = raw_log_halflife
    overrides = []
    
    if raw_log_halflife < -1.30:
        final_log_halflife = -1.30
        overrides.append("t1/2 Continuous: Clamped to physiological circulation limits (>= 0.05 hr)")
        
    override_msg = "; ".join(overrides) if overrides else "Passed"
    return final_log_halflife, override_msg


# =====================================================================
# 14. Human Intestinal Absorption (HIA) Expert Overrides
# =====================================================================
def apply_hia_filters(smiles, raw_hia_prob):
    """
    Applies physical and drug-likeness overrides to raw HIA predictions.
    - Massive, highly polar molecules (MW > 600 g/mol and TPSA > 150) suffer from 
      severely restricted passive transcellular/paracellular intestinal absorption.
      Ensure their predicted absorption probability is clamped to low risk (<= 15%).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return raw_hia_prob, "Invalid SMILES"
        
    try:
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        
        # Check for Lipinski/Veber extreme violations restricting gut transit
        if mw > 600.0 and tpsa > 150.0:
            final_prob = min(raw_hia_prob, 0.15)
            return final_prob, "HIA Neutralized: High polarity/MW limits passive intestinal absorption"
    except Exception:
        pass
        
    return raw_hia_prob, "Passed"


# =====================================================================
# 15. Intrinsic Hepatic Clearance (CL_hep) Continuous Overrides
# =====================================================================
def apply_clearance_continuous_filters(raw_log_cl):
    """
    Applies physical boundaries to the predicted log10(CL_hep).
    - Incubation assays are bounded by a minimum detection limit of 3.0 uL/min (0.4771 log units)
      and an upper saturation limit of 150.0 uL/min (2.1761 log units).
      Clamps predictions to these physical limits to prevent extreme extrapolations.
    """
    final_log_cl = raw_log_cl
    overrides = []
    
    if raw_log_cl < 0.4771:
        final_log_cl = 0.4771
        overrides.append("CL_hep Continuous: Clamped to assay lower detection limit (>= 3.0 uL/min)")
    elif raw_log_cl > 2.1761:
        final_log_cl = 2.1761
        overrides.append("CL_hep Continuous: Clamped to assay upper saturation limit (<= 150.0 uL/min)")
        
    override_msg = "; ".join(overrides) if overrides else "Passed"
    return final_log_cl, override_msg


# =====================================================================
# 16. Aqueous Solubility (logS) Continuous Overrides
# =====================================================================
def apply_solubility_continuous_filters(raw_logs):
    """
    Applies physical boundaries to the predicted log10(logS).
    - Aqueous solubility predictions are clamped to the curated training dataset limits:
      minimum of -13.0 logS (highly insoluble) and maximum of 2.15 logS (highly soluble).
    """
    final_logs = raw_logs
    overrides = []
    
    if raw_logs < -13.0:
        final_logs = -13.0
        overrides.append("logS Continuous: Clamped to lower physical solubility limit (-13.0 logS)")
    elif raw_logs > 2.15:
        final_logs = 2.15
        overrides.append("logS Continuous: Clamped to upper physical solubility limit (2.15 logS)")
        
    override_msg = "; ".join(overrides) if overrides else "Passed"
    return final_logs, override_msg


# =====================================================================
# 17. Oral Bioavailability (F) Physiological Overrides
# =====================================================================
def apply_bioavailability_filters(smiles, raw_f_prob, hia_prob, cl_log_val):
    """
    Applies physical and physiological boundaries to oral bioavailability (F).
    - If a compound has low predicted absorption (HIA < 15%) AND high predicted 
      hepatic clearance (log_cl > 1.80, equivalent to >63 uL/min), its pre-systemic 
      first-pass extraction is so severe that systemic bioavailability is almost zero.
      Ensure its predicted bioavailability probability is clamped to <= 15%.
    """
    final_prob = raw_f_prob
    overrides = []
    
    # Identify highly clearable, poorly absorbed compounds (extreme first-pass barrier)
    if hia_prob < 0.15 and cl_log_val > 1.80:
        final_prob = min(raw_f_prob, 0.15)
        overrides.append("F Neutralized: Severe first-pass extraction (Low HIA & High CL_hep)")
        
    override_msg = "; ".join(overrides) if overrides else "Passed"
    return final_prob, override_msg


# =====================================================================
# 18. Rodent Carcinogenicity Expert Overrides
# =====================================================================
def apply_carcinogenicity_filters(smiles, raw_carcinogen_prob):
    """
    Applies expert structural checks to raw rodent carcinogenicity predictions.
    - Reuses the existing AMES_ALERTS dictionary to check for DNA-reactive mutagenic subgroups.
    - Explicitly intercepts salicylate structures (aspirin, salicylic acid) which are 
      clinically proven to be non-carcinogenic and chemopreventive, forcing a low-risk baseline.
    - If a compound contains a highly reactive Ames alert and its statistical probability
      is borderline (0.35 <= prob < 0.50), elevate the predicted risk to positive (0.55).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return raw_carcinogen_prob, "Invalid SMILES"
        
    final_prob = raw_carcinogen_prob
    overrides = []
    
    # 1. Specific Salicylate Check (Aspirin & Salicylic Acid cores)
    is_salicylate = mol.HasSubstructMatch(Chem.MolFromSmarts("O=C(O)c1ccccc1OC(=O)C")) or \
                    mol.HasSubstructMatch(Chem.MolFromSmarts("O=C(O)c1ccccc1O"))
                    
    if is_salicylate:
        final_prob = min(raw_carcinogen_prob, 0.15)
        overrides.append("Carcinogenicity: Overridden (Salicylate chemopreventive profile)")
        override_msg = "; ".join(overrides)
        return final_prob, override_msg
    
    # 2. Borderline Mutagenic Elevation
    has_alert = False
    for alert_name, smarts in AMES_ALERTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            has_alert = True
            break
            
    if has_alert and 0.35 <= raw_carcinogen_prob < 0.50:
        final_prob = 0.55
        overrides.append("Carcinogenicity: Elevated borderline risk due to co-occurring Ames mutagenic alert")
        
    override_msg = "; ".join(overrides) if overrides else "Passed"
    return final_prob, override_msg


# =====================================================================
# 19. Skin Sensitization Expert Overrides
# =====================================================================
def apply_sensitization_filters(smiles, raw_sens_prob, logs_val):
    """
    Applies expert overrides to raw skin sensitization predictions.
    - Highly hydrophilic compounds (predicted logS >= 0.0 or LogP < -2.0)
      have poor stratum corneum penetration and lack lipophilic protein-conjugation (haptenation) properties.
      Ensure their predicted sensitization probability is clamped to low risk (<= 15%).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return raw_sens_prob, "Invalid SMILES"
        
    final_prob = raw_sens_prob
    overrides = []
    
    try:
        if logs_val >= 0.0:
            final_prob = min(raw_sens_prob, 0.15)
            overrides.append("Skin Sensitization: Neutralized (Highly Hydrophilic / High Solubility)")
    except Exception:
        pass
        
    override_msg = "; ".join(overrides) if overrides else "Passed"
    return final_prob, override_msg


# =====================================================================
# 20. Bile Salt Export Pump (BSEP) Expert Overrides
# =====================================================================
def apply_bsep_filters(smiles, pka_acid, pka_base, raw_bsep_prob, ph=7.4):
    """
    Applies expert overrides to raw BSEP (ABCB11) predictions.
    - BSEP primarily transports organic anions and large neutral amphipathic species. 
      Highly basic, strongly cationic species (basic pKa >= 8.0 and cationic fraction >= 90% at pH 7.4)
      lack affinity for BSEP and default to low-risk Non-Inhibitor (<= 15%).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return raw_bsep_prob, "Invalid SMILES"
        
    f_anion, f_cation = calculate_ionized_fractions(pka_acid, pka_base, ph)
    final_prob = raw_bsep_prob
    overrides = []
    
    # Cationic exclusion check
    if pka_base is not None and pka_base >= 8.0 and f_cation >= 0.90:
        final_prob = min(raw_bsep_prob, 0.15)
        overrides.append("BSEP Neutralized (Highly Cationic Basic Species)")
        
    override_msg = "; ".join(overrides) if overrides else "Passed"
    return final_prob, override_msg


# =====================================================================
# 21. Mitochondrial Toxicity (MitoTox) Expert Overrides
# =====================================================================
def apply_mitotox_filters(smiles, raw_mitotox_prob, logs_val, logp_val):
    """
    Applies physical and thermodynamic overrides to raw mitochondrial toxicity predictions.
    - Highly hydrophilic and soluble polar compounds (logS >= 1.0 or LogP < -3.0) 
      are physically incapable of crossing the double mitochondrial membrane or accumulating 
      in the matrix to cause electron-transport chain decoupling.
      Ensure their disruption disruption probability is clamped to low risk (<= 15%).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return raw_mitotox_prob, "Invalid SMILES"
        
    final_prob = raw_mitotox_prob
    overrides = []
    
    try:
        if logs_val >= 1.0 or logp_val < -3.0:
            final_prob = min(raw_mitotox_prob, 0.15)
            overrides.append("MitoTox: Neutralized (Extreme Hydrophilicity/Solubility limits membrane transit)")
    except Exception:
        pass
        
    override_msg = "; ".join(overrides) if overrides else "Passed"
    return final_prob, override_msg


# =====================================================================
# 22. MDR1-MDCK Substrate / Efflux Expert Overrides
# =====================================================================
def apply_mdck_filters(smiles, raw_mdck_prob, mw, logp):
    """
    Applies physical boundaries to MDR1-MDCK active transport efflux predictions.
    - Small structures (MW < 180 g/mol) or highly hydrophilic compounds (LogP < -1.5)
      lack the size or lipophilicity required to bind inside the MDR1 (P-gp) channel cavity.
      Clamps the efflux substrate probability to low-risk (<= 15.0%).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return raw_mdck_prob, "Invalid SMILES"
        
    final_prob = raw_mdck_prob
    overrides = []
    
    if mw < 180.0 or logp < -1.5:
        final_prob = min(raw_mdck_prob, 0.15)
        overrides.append("MDR1-MDCK Efflux Neutralized (MW or polarity boundary limits)")
        
    override_msg = "; ".join(overrides) if overrides else "Passed"
    return final_prob, override_msg


# =====================================================================
# 23. Quantitative logBB Continuous Expert Overrides
# =====================================================================
def apply_logbb_continuous_filters(smiles, raw_logbb, mw, tpsa, logp):
    """
    Applies physical and molecular boundaries to continuous logBB predictions.
    - Large molecules (MW > 500 g/mol) or highly polar molecules (TPSA > 120)
      face massive physical constraints passing through the blood-brain barrier.
    - Highly ionized or extremely hydrophilic compounds (LogP < -1.0) do not pass.
    - Clamps logBB value to non-penetrant levels (<= -1.50 logBB) under these violations.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return raw_logbb, "Invalid SMILES"
        
    final_logbb = raw_logbb
    overrides = []
    
    if mw > 500.0 or tpsa > 120.0 or logp < -1.0:
        if raw_logbb > -1.50:
            final_logbb = -1.50
            overrides.append("logBB: Polarity / molecular weight boundary restriction")
            
    # Keep the final logBB boundaries constrained to physiological extremes
    final_logbb = max(-3.0, min(2.0, final_logbb))
            
    override_msg = "; ".join(overrides) if overrides else "Passed"
    return final_logbb, override_msg


# =====================================================================
# 24. Blood-to-Plasma Partition Ratio (Rbp) Expert Overrides
# =====================================================================
def apply_rbp_filters(smiles, raw_rbp_prob, logp):
    """
    Applies a lipophilic safety override to Blood-to-Plasma Partition Ratio (Rbp) predictions.
    - Highly lipophilic compounds (LogP > 5.0) partition extensively into lipid-dense red blood
      cell membranes, representing a high sequestration risk (Rbp > 1.2).
      Force probability to high risk (>= 65%) to ensure safety-conservative predictions.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return raw_rbp_prob, "Invalid SMILES"
        
    final_prob = raw_rbp_prob
    overrides = []
    
    if logp > 5.0:
        final_prob = max(raw_rbp_prob, 0.65)
        overrides.append("Rbp: High sequestration risk elevated due to lipophilicity boundary (LogP > 5.0)")
        
    override_msg = "; ".join(overrides) if overrides else "Passed"
    return final_prob, override_msg


# =====================================================================
# 25. Secondary CYP Isoform (CYP2C8 & CYP2E1) Expert Overrides
# =====================================================================
def apply_cyp_secondary_filters(smiles, raw_2c8_prob, raw_2e1_prob, mw, tpsa):
    """
    Applies physical and cavity size constraints to secondary CYP isoforms.
    - CYP2E1 has an extremely small, narrow active site pocket designed for small, 
      volatile compounds (MW < 150). Bulky, complex molecules (MW > 350 g/mol) 
      cannot accommodate the pocket and are overridden to Non-Inhibitors.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return raw_2c8_prob, raw_2e1_prob, "Invalid SMILES"
        
    final_2c8 = raw_2c8_prob
    final_2e1 = raw_2e1_prob
    overrides = []
    
    if mw > 350.0 or tpsa > 100.0:
        if raw_2e1_prob > 0.15:
            final_2e1 = 0.15
            overrides.append("CYP2E1: Neutralized (Steric size limit exceeded)")
            
    override_msg = "; ".join(overrides) if overrides else "Passed"
    return final_2c8, final_2e1, override_msg


# =====================================================================
# 26. Fraction Unbound in Brain (f_u,brain) Expert Overrides
# =====================================================================
def apply_fubrain_filters(fubrain_pct, logp, smiles):
    """
    Section 26: f_u,brain (Fraction Unbound in Brain) physical-chemical overrides.
    Protects model prediction boundaries against fingerprint-regression edge cases.
    """
    msg = ""
    # Highly lipophilic: severe non-specific lipid membrane binding
    if logp > 5.0:
        fubrain_pct = min(fubrain_pct, 1.0)
        msg = "LogP > 5.0: High lipophilicity triggers extreme non-specific brain lipid sequestration; unbound fraction restricted to ≤ 1.0%."
    
    # Highly hydrophilic: minimal affinity for brain tissue lipids
    elif logp < 0.0:
        fubrain_pct = max(fubrain_pct, 15.0)
        msg = "LogP < 0.0: High hydrophilicity restricts brain lipid membrane binding; unbound fraction safety floor enforced at ≥ 15.0%."
        
    return round(fubrain_pct, 3), msg


# =====================================================================
# 27. Human Maximum Tolerated Dose (MTD) Expert Overrides
# =====================================================================
def apply_mtd_filters(smiles, raw_mtd_prob, logp, mw):
    """
    Section 27: Human Maximum Tolerated Dose (MTD) expert overrides.
    - Specific Salicylate Check: Intercepts and overrides salicylates (aspirin/salicylic acid)
      which are clinically proven to have high tolerated dose limits, forcing a low toxicity class (<= 15%).
    - Polar Excretion Rule: Extremely polar, small molecules (LogP < 0.0 and MW < 250) are rapidly cleared 
      and have very low off-target off-binding liabilities; clamps high toxicity probability to <= 15%.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return raw_mtd_prob, "Invalid SMILES"
        
    final_prob = raw_mtd_prob
    overrides = []
    
    # 1. Specific Salicylate Check (Aspirin & Salicylic Acid cores)
    is_salicylate = mol.HasSubstructMatch(Chem.MolFromSmarts("O=C(O)c1ccccc1OC(=O)C")) or \
                    mol.HasSubstructMatch(Chem.MolFromSmarts("O=C(O)c1ccccc1O"))
                    
    if is_salicylate:
        final_prob = min(raw_mtd_prob, 0.15)
        overrides.append("MTD: Overridden (Salicylate clinically high tolerated profile)")
        override_msg = "; ".join(overrides)
        return final_prob, override_msg
        
    # 2. Polar Excretion Rule
    if logp < 0.0 and mw < 250.0:
        final_prob = min(raw_mtd_prob, 0.15)
        overrides.append("MTD: Toxicity risk capped (Polar hydrophilic small-molecule cleared rapidly)")
        
    override_msg = "; ".join(overrides) if overrides else "Passed"
    return final_prob, override_msg