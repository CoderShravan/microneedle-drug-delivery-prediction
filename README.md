# TransDerm.AI — Microneedle Drug Delivery & Skin PK Predictor

An interactive, natural, light-themed web application and machine learning prediction system for transdermal microneedle drug delivery kinetics.

Based on the research and multi-output ML models trained across 150 microneedle patch formulations in `microneedle_concentration_time_ML.ipynb`.

---

## Key Features

1. **Exact Machine Learning Pharmacokinetic Model**:
   - Reconstructs the validated **Multi-Output Log1p Lasso Model (Strategy 2)** from the notebook (`R² = 0.841`, test set verified).
   - Standardizes all numerical features using the empirical training distributions (`StandardScaler`).
   - One-hot encodes drug categories, microneedle materials, and skin substrates.
   - Computes out-of-fold 80% confidence interval bands ($q_{10}$ to $q_{90}$).
   - Generates monotonic PCHIP spline continuous curves from $0.5$ h to $24.0$ h.

2. **Required Feature Inputs**:
   - **Drug Formulation**: Lidocaine, Caffeine, Metformin, Doxorubicin, Insulin, or Custom Drug (Molecular Weight in Da, $\log P$, Large Peptide flag).
   - **Patch Architecture**: Initial Drug Loading ($1.03 - 9.76$ mg), Patch Area ($0.52 - 2.49\text{ cm}^2$), Needle Count ($100, 144, 196, 256$), Needle Height ($507.6 - 797.0\ \mu\text{m}$), Base Width ($211.7 - 349.5\ \mu\text{m}$), Applied Force ($15.3 - 34.8\text{ N}$).
   - **Polymer & Substrate**: Hyaluronic Acid (Dissolving), PVA / PVP (Dissolving), PLGA 50:50 (Solid Polymeric), Gantrez S-97 / PEG (Hydrogel-Forming).
   - **Skin Model**: Human Skin (Full Thickness — calibrated), Porcine Skin, Rat Skin, Synthesised Polymeric Membrane, Skin Thickness ($0.81 - 1.77\text{ mm}$).

3. **Pharmacokinetic Outputs & Predictions**:
   - **$C_{\max}$**: Peak skin tissue concentration ($\mu\text{g}/\text{cm}^3$).
   - **$T_{\max}$**: Time to reach peak concentration (hours).
   - **$\text{AUC}_{0.5-24\text{h}}$**: Total cumulative tissue exposure ($\mu\text{g}\cdot\text{h}/\text{cm}^3$).
   - **Drug Consumed / Retained in Skin**: Tissue volume calculation and total micrograms ($\mu\text{g}$) delivered into viable skin.
   - **Delivery Efficiency**: Percentage of initially loaded patch dose actively residing in skin tissue.
   - **5 Key Time Interval Table**: Discrete breakdown at $1.0\text{ h}, 2.0\text{ h}, 4.0\text{ h}, 8.0\text{ h}, 24.0\text{ h}$ with credible intervals and absorption progress.

4. **Natural & Light Visual Design**:
   - Inspired by modern, organic, light-mode user interfaces.
   - Warm cream canvas (`#FAF8F5`) with subtle radial ambient glow.
   - Soft rounded cards (`border-radius: 24px`), delicate borders, and pillowy shadows.
   - Violet-to-peach gradient accent buttons and clean typography (`Plus Jakarta Sans`).
   - Live interactive scrubber over the concentration-time curve.

---

## How to Run

### Option 1: Direct in Browser (Instant, Zero Setup)
Simply double-click or open `microneedle_app/index.html` in any web browser (Chrome, Edge, Safari, Firefox). The entire model and dynamic visualization run client-side with 0ms latency!

### Option 2: Local Python Server & API
Run the Python HTTP server to serve the app and provide the `/api/predict` endpoint:

```powershell
python microneedle_app/server.py
```

Then visit:
```
http://localhost:8501
```

---

## File Structure

- `index.html` — The standalone single-page responsive UI application.
- `mn_model.py` — The standalone Python ML prediction module with validation and PCHIP spline interpolation.
- `server.py` — Lightweight Python server providing REST endpoints (`/api/predict`, `/api/info`).
- `README.md` — Project documentation and usage guide.
