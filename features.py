# features.py
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import DataStructs

def calculate_hybrid_features(smiles):
    """
    Generates a 1,035-dimensional hybrid feature vector.
    - 11 physicochemical descriptors
    - 1,024-bit Morgan fingerprint (radius=2)
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    try:
        # Standardize and sanitize molecule representation
        Chem.SanitizeMol(mol)
    except Exception:
        return None
        
    try:
        # Extract 11 molecular descriptors
        phys_features = [
            float(Descriptors.MolWt(mol)),
            float(Descriptors.MolLogP(mol)),
            float(Descriptors.TPSA(mol)),
            float(Descriptors.NumHAcceptors(mol)),
            float(Descriptors.NumHDonors(mol)),
            float(Descriptors.NumRotatableBonds(mol)),
            float(Descriptors.FractionCSP3(mol)),
            float(Descriptors.HeavyAtomCount(mol)),
            float(Descriptors.RingCount(mol)),
            float(Descriptors.MolMR(mol)),
            float(Descriptors.NumValenceElectrons(mol))
        ]
        
        # Guard against NaN/Inf values
        phys_features = [0.0 if np.isnan(x) or np.isinf(x) else x for x in phys_features]
        phys_array = np.array(phys_features, dtype=float)
        
        # Calculate 1,024-bit topological Morgan fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fp_array = np.zeros((1024,), dtype=float)
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        
        # Concatenate features into a single array (11 + 1024 = 1035 features)
        hybrid_vector = np.concatenate([phys_array, fp_array])
        return hybrid_vector
    except Exception:
        return None

if __name__ == "__main__":
    # Test execution with Aspirin
    test_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    print("Testing feature extractor with Aspirin...")
    vector = calculate_hybrid_features(test_smiles)
    if vector is not None:
        print(f"Extraction status: Successful.")
        print(f"Generated vector shape: {vector.shape}")
        if vector.shape[0] == 1035:
            print("Verification: Feature dimensions match the 1,035 target.")
        else:
            print(f"Verification Failure: Vector has {vector.shape[0]} dimensions instead of 1,035.")
    else:
        print("Extraction status: Failed. Please verify your RDKit environment.")