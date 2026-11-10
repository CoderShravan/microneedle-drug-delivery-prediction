"""
mn_model.py
Standalone Microneedle Concentration-Time Prediction Model.
Reconstructed from microneedle_concentration_time_ML.ipynb (Multi-output log1p Lasso model, Strategy 2).
"""

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.integrate import trapezoid

# Standard evaluation time points in hours
TIME_POINTS = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0])

# Known drug library
DRUGS_DB = {
    'Lidocaine': {
        'Drug_MW_Da': 234.34,
        'Drug_logP': 2.44,
        'Is_Large_Peptide': 0,
        'description': 'Amide local anesthetic for targeted dermal anesthesia'
    },
    'Caffeine': {
        'Drug_MW_Da': 194.19,
        'Drug_logP': -0.07,
        'Is_Large_Peptide': 0,
        'description': 'Model small hydrophilic alkaloid'
    },
    'Metformin': {
        'Drug_MW_Da': 129.16,
        'Drug_logP': -1.40,
        'Is_Large_Peptide': 0,
        'description': 'Hydrophilic biguanide anti-diabetic agent'
    },
    'Doxorubicin': {
        'Drug_MW_Da': 543.52,
        'Drug_logP': 1.27,
        'Is_Large_Peptide': 0,
        'description': 'Anthracycline antineoplastic chemotherapy drug'
    },
    'Insulin': {
        'Drug_MW_Da': 5808.00,
        'Drug_logP': -1.80,
        'Is_Large_Peptide': 1,
        'description': 'Large 51-amino acid therapeutic peptide'
    }
}

MATERIAL_TO_TYPE = {
    'Hyaluronic Acid': 'Dissolving',
    'PVA / PVP': 'Dissolving',
    'PLGA 50:50': 'Solid Polymeric',
    'Gantrez S-97 / PEG': 'Hydrogel-Forming'
}

SKIN_TYPES = [
    'Human Skin (Full Thickness)',
    'Porcine Skin',
    'Rat Skin',
    'Synthesised Polymeric Skin'
]

# Training bounds for numeric features
NUMERIC_RANGES = {
    'Skin_Thickness_mm': (0.81, 1.77),
    'Patch_Area_cm2': (0.52, 2.49),
    'Needle_Count': (100, 256),
    'Needle_Height_um': (507.6, 797.0),
    'Needle_Base_Width_um': (211.7, 349.5),
    'Applied_Force_N': (15.3, 34.8),
    'Initial_Drug_Loading_mg': (1.03, 9.76),
    'Drug_MW_Da': (129.16, 5808.00),
    'Drug_logP': (-1.80, 2.44)
}

# StandardScaler parameters (mean, std) from the 150 training patches
SCALER_STATS = {
    'Skin_Thickness_mm': (1.300, 0.270),
    'Patch_Area_cm2': (1.478, 0.622),
    'Needle_Count': (170.693, 50.905),
    'Needle_Height_um': (661.214, 82.644),
    'Needle_Base_Width_um': (277.531, 40.814),
    'Applied_Force_N': (24.371, 5.761),
    'Initial_Drug_Loading_mg': (5.507, 2.460),
    'Drug_MW_Da': (1304.641, 2162.768),
    'Drug_logP': (0.219, 1.646),
    'Is_Large_Peptide': (0.187, 0.390),
}

# Exact coefficients for the 24 non-zero features at the 8 time points:
# Columns: [0.5h, 1.0h, 2.0h, 4.0h, 6.0h, 8.0h, 12.0h, 24.0h]
COEFFICIENTS = {
    'Drug_Name_Doxorubicin': [-1.11, -1.02, -1.08, -0.73, -0.39, -0.17, -0.04, 0.27],
    'Drug_Name_Metformin': [-0.58, -0.63, -0.76, -0.54, -0.19, -0.00, 0.00, 0.22],
    'Initial_Drug_Loading_mg': [0.38, 0.41, 0.47, 0.50, 0.53, 0.53, 0.53, 0.50],
    'MN_Type_Dissolving': [-0.00, 0.00, 0.00, 0.00, -0.00, 0.05, 0.07, 0.40],
    'MN_Material_Hyaluronic Acid': [0.19, 0.20, 0.20, 0.28, 0.33, 0.32, 0.39, 0.40],
    'MN_Material_PVA / PVP': [-0.18, -0.22, -0.32, -0.20, -0.06, 0.00, -0.00, -0.00],
    'Skin_Type_Rat Skin': [0.05, 0.04, 0.06, -0.01, -0.05, -0.06, -0.09, -0.26],
    'Drug_Name_Caffeine': [0.00, 0.13, 0.16, 0.21, 0.25, 0.25, 0.16, -0.00],
    'Is_Large_Peptide': [0.05, 0.00, 0.00, 0.00, 0.00, -0.00, -0.00, -0.21],
    'Drug_logP': [0.19, 0.20, 0.19, 0.10, 0.08, 0.06, -0.04, -0.13],
    'Patch_Area_cm2': [0.02, 0.02, 0.05, -0.04, -0.11, -0.13, -0.15, -0.13],
    'Needle_Count': [-0.04, -0.01, -0.01, -0.01, 0.03, 0.05, 0.05, 0.13],
    'Applied_Force_N': [0.08, 0.02, 0.03, 0.02, 0.04, 0.05, 0.03, 0.13],
    'MN_Material_PLGA 50:50': [0.12, 0.06, 0.11, 0.00, 0.00, -0.00, -0.10, 0.00],
    'Skin_Type_Synthesised Polymeric Skin': [-0.06, -0.12, -0.03, 0.03, 0.06, 0.04, 0.03, -0.00],
    'Skin_Type_Porcine Skin': [0.11, -0.00, 0.07, 0.00, 0.00, -0.01, -0.00, 0.10],
    'Skin_Type_Human Skin (Full Thickness)': [-0.09, 0.00, -0.08, -0.06, -0.00, 0.03, 0.01, 0.10],
    'Drug_MW_Da': [0.00, 0.00, 0.00, 0.00, 0.00, -0.01, -0.10, -0.00],
    'MN_Material_Gantrez S-97 / PEG': [-0.00, -0.00, -0.00, -0.00, -0.00, -0.01, 0.00, -0.08],
    'Needle_Height_um': [0.06, 0.06, 0.07, 0.06, 0.04, 0.02, -0.02, -0.02],
    'MN_Type_Solid Polymeric': [0.06, 0.05, 0.04, 0.00, 0.02, -0.00, -0.00, 0.00],
    'Needle_Base_Width_um': [0.03, 0.04, 0.04, 0.00, -0.01, -0.04, -0.05, -0.02],
    'Skin_Thickness_mm': [0.00, 0.01, 0.01, 0.02, 0.00, -0.01, 0.02, 0.04],
    'MN_Type_Hydrogel-Forming': [-0.03, -0.03, -0.03, -0.00, -0.00, 0.00, 0.00, -0.00],
}

INTERCEPTS = np.array([2.0956, 2.8704, 3.6150, 4.0343, 3.8110, 3.5227, 3.1225, 1.7216])

# 80% out-of-fold residual quantiles (q10, q90) for log1p predictions
RESID_Q10 = np.array([-0.7398, -0.6690, -0.6961, -0.4198, -0.3862, -0.4559, -0.5528, -0.8530])
RESID_Q90 = np.array([0.7365, 0.7338, 0.7233, 0.4890, 0.3893, 0.4587, 0.5214, 1.0571])


def validate_inputs(inp):
    """Checks input ranges and formulation consistency, returning friendly warnings."""
    warnings = []
    
    # Range checks
    for key, (lo, hi) in NUMERIC_RANGES.items():
        if key in inp:
            val = float(inp[key])
            if val < lo or val > hi:
                warnings.append({
                    'type': 'range',
                    'field': key,
                    'message': f"{key.replace('_', ' ')} = {val} is outside the validated training range [{lo}, {hi}]."
                })

    # Drug check
    drug = inp.get('drug_name')
    if drug in DRUGS_DB:
        db_drug = DRUGS_DB[drug]
        if abs(float(inp.get('drug_mw', db_drug['Drug_MW_Da'])) - db_drug['Drug_MW_Da']) / db_drug['Drug_MW_Da'] > 0.05:
            warnings.append({
                'type': 'drug_property',
                'field': 'drug_mw',
                'message': f"MW deviates from reference {drug} ({db_drug['Drug_MW_Da']} Da)."
            })
    else:
        warnings.append({
            'type': 'unseen_drug',
            'field': 'drug_name',
            'message': f"Drug '{drug}' is novel/unseen in training data. Predictions should be interpreted with caution."
        })

    # Human skin calibration note
    if str(inp.get('skin_type', '')).startswith('Human'):
        warnings.append({
            'type': 'human_skin',
            'field': 'skin_type',
            'message': "Human skin calibration rests on 27 full-thickness human patches; investigational use only."
        })

    return warnings


def predict_profile(
    drug_name='Lidocaine',
    drug_mw=234.34,
    drug_logp=2.44,
    initial_drug_loading_mg=6.0,
    mn_material='Hyaluronic Acid',
    mn_type='Dissolving',
    needle_count=196,
    needle_height_um=650.0,
    needle_base_width_um=280.0,
    patch_area_cm2=1.5,
    applied_force_n=25.0,
    skin_type='Human Skin (Full Thickness)',
    skin_thickness_mm=1.25,
    custom_times=None
):
    """
    Predict skin-tissue concentration vs time, Cmax, Tmax, AUC, and drug consumed/delivered.
    """
    # Material-to-type alignment
    if mn_material in MATERIAL_TO_TYPE:
        mn_type = MATERIAL_TO_TYPE[mn_material]

    is_peptide = 1 if float(drug_mw) > 5000 else 0

    inp = {
        'drug_name': drug_name,
        'drug_mw': float(drug_mw),
        'drug_logp': float(drug_logp),
        'initial_drug_loading_mg': float(initial_drug_loading_mg),
        'mn_material': mn_material,
        'mn_type': mn_type,
        'needle_count': float(needle_count),
        'needle_height_um': float(needle_height_um),
        'needle_base_width_um': float(needle_base_width_um),
        'patch_area_cm2': float(patch_area_cm2),
        'applied_force_n': float(applied_force_n),
        'skin_type': skin_type,
        'skin_thickness_mm': float(skin_thickness_mm),
        'is_large_peptide': is_peptide
    }

    warnings = validate_inputs(inp)

    # Standardize numerical features
    num_vals = {
        'Skin_Thickness_mm': inp['skin_thickness_mm'],
        'Patch_Area_cm2': inp['patch_area_cm2'],
        'Needle_Count': inp['needle_count'],
        'Needle_Height_um': inp['needle_height_um'],
        'Needle_Base_Width_um': inp['needle_base_width_um'],
        'Applied_Force_N': inp['applied_force_n'],
        'Initial_Drug_Loading_mg': inp['initial_drug_loading_mg'],
        'Drug_MW_Da': inp['drug_mw'],
        'Drug_logP': inp['drug_logp'],
        'Is_Large_Peptide': inp['is_large_peptide']
    }

    feature_vector = {}
    for col, (mean, std) in SCALER_STATS.items():
        feature_vector[col] = (num_vals[col] - mean) / std

    # One-hot encoded indicators
    feature_vector['Drug_Name_' + drug_name] = 1.0
    feature_vector['Skin_Type_' + skin_type] = 1.0
    feature_vector['MN_Material_' + mn_material] = 1.0
    feature_vector['MN_Type_' + mn_type] = 1.0

    # Linear combination
    linear_pred = np.zeros(8)
    for feat_name, coefs in COEFFICIENTS.items():
        v = feature_vector.get(feat_name, 0.0)
        linear_pred += v * np.array(coefs)

    # Apply target log1p inverse transform
    log1p_target = linear_pred + INTERCEPTS
    predicted_concs = np.clip(np.expm1(log1p_target), 0.0, None)

    # 80% Confidence bounds
    lower_80 = np.clip(np.expm1(log1p_target - RESID_Q90), 0.0, None)
    upper_80 = np.clip(np.expm1(log1p_target - RESID_Q10), 0.0, None)

    # Monotonic PCHIP continuous spline for smooth profile
    pchip_pred = PchipInterpolator(TIME_POINTS, predicted_concs)
    pchip_lower = PchipInterpolator(TIME_POINTS, lower_80)
    pchip_upper = PchipInterpolator(TIME_POINTS, upper_80)

    # Dense time curve for high-resolution plotting (0.5 to 24h, 150 points)
    dense_times = np.linspace(0.5, 24.0, 150)
    dense_pred = np.clip(pchip_pred(dense_times), 0.0, None)
    dense_lower = np.clip(pchip_lower(dense_times), 0.0, None)
    dense_upper = np.clip(pchip_upper(dense_times), 0.0, None)

    # Peak metrics
    max_idx = int(np.argmax(predicted_concs))
    cmax = float(predicted_concs[max_idx])
    tmax = float(TIME_POINTS[max_idx])
    auc_total = float(trapezoid(predicted_concs, TIME_POINTS))

    # Skin tissue volume in cm3: Patch_Area (cm2) * Skin_Thickness (converted to cm = mm * 0.1)
    tissue_vol_cm3 = inp['patch_area_cm2'] * (inp['skin_thickness_mm'] * 0.1)
    
    # Calculate drug amount in skin (ug) = Conc (ug/cm3) * Tissue Volume (cm3)
    drug_amount_ug = predicted_concs * tissue_vol_cm3
    
    # Cumulative drug delivered/absorbed: estimated from integral or dynamic clearance
    # Peak drug inside skin tissue:
    peak_drug_in_skin_ug = cmax * tissue_vol_cm3
    initial_dose_ug = inp['initial_drug_loading_mg'] * 1000.0
    delivery_efficiency_pct = (peak_drug_in_skin_ug / initial_dose_ug) * 100.0 if initial_dose_ug > 0 else 0.0

    # Standard time points table
    profile_table = []
    for t, c, lo, hi, d_amt in zip(TIME_POINTS, predicted_concs, lower_80, upper_80, drug_amount_ug):
        profile_table.append({
            'time_hours': float(t),
            'conc_ug_cm3': round(float(c), 2),
            'lower_80_ug_cm3': round(float(lo), 2),
            'upper_80_ug_cm3': round(float(hi), 2),
            'drug_in_skin_ug': round(float(d_amt), 2),
            'pct_loading_in_skin': round(float(d_amt / initial_dose_ug * 100.0), 3)
        })

    # Custom 4-5 inspection time points (if requested by user)
    if custom_times is None:
        custom_times = [1.0, 2.0, 4.0, 8.0, 24.0]
    
    custom_points_table = []
    for ct in custom_times:
        c_val = float(np.clip(pchip_pred(ct), 0.0, None))
        lo_val = float(np.clip(pchip_lower(ct), 0.0, None))
        hi_val = float(np.clip(pchip_upper(ct), 0.0, None))
        d_val = c_val * tissue_vol_cm3
        custom_points_table.append({
            'time_hours': float(ct),
            'conc_ug_cm3': round(c_val, 2),
            'lower_80_ug_cm3': round(lo_val, 2),
            'upper_80_ug_cm3': round(hi_val, 2),
            'drug_in_skin_ug': round(d_val, 2),
            'pct_loading_in_skin': round(float(d_val / initial_dose_ug * 100.0), 3)
        })

    return {
        'inputs': inp,
        'summary': {
            'cmax_ug_cm3': round(cmax, 2),
            'tmax_hours': round(tmax, 1),
            'auc_ug_h_cm3': round(auc_total, 2),
            'peak_drug_in_skin_ug': round(peak_drug_in_skin_ug, 2),
            'delivery_efficiency_pct': round(delivery_efficiency_pct, 2),
            'tissue_volume_cm3': round(tissue_vol_cm3, 4),
            'initial_dose_ug': round(initial_dose_ug, 1)
        },
        'warnings': warnings,
        'standard_profile': profile_table,
        'custom_profile': custom_points_table,
        'dense_curve': {
            'times': [round(float(t), 2) for t in dense_times],
            'conc': [round(float(c), 2) for c in dense_pred],
            'lower': [round(float(lo), 2) for lo in dense_lower],
            'upper': [round(float(hi), 2) for hi in dense_upper]
        }
    }


if __name__ == '__main__':
    # Quick self-test
    res = predict_profile()
    print("Self test Cmax:", res['summary']['cmax_ug_cm3'], "Tmax:", res['summary']['tmax_hours'])
    print("4-5 Inspection Times:")
    for pt in res['custom_profile']:
        print(f"  {pt['time_hours']}h -> {pt['conc_ug_cm3']} ug/cm3 | Drug in skin: {pt['drug_in_skin_ug']} ug")
