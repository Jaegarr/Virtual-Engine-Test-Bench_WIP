# 🧪 Digital Twin Proof-of-Concept Demonstrations

---

## 🎯 Overview
This repository contains three proof-of-concept demonstrations and show the author’s prior independent work towards an integrated **multi-fidelity digital twin** architecture connecting:
- physics-based combustion modelling,  
- machine-learning corrections from dyno data, and  
- CFD coupling.

All simulations were performed using the author’s existing **Virtual Engine Test Bench (VETB)** Python framework.

---

## 🔹 Demo 1 — Multi-Fuel Combustion Behaviour

**Objective:**  
Evaluate the combustion characteristics of **gasoline, hydrogen, ammonia**, and a **10 % H₂–NH₃ blend** at 3000 rpm under wide-open-throttle (WOT) conditions to assess lean-limit behaviour and fuel-specific IMEP trends.

**Methodology:**
- Single-zone Wiebe-function combustion model (Python).  
- Stoichiometric to lean operation (λ = 0.8 – 1.5).  
- Fixed burn duration beyond λ ≥ 1.3 for numerical stability.  
- Fuel-specific thermochemical and laminar flame-speed correlations.  

**Results Summary:**
- Hydrogen achieved the highest IMEP (~15.6 bar at λ = 0.8) and stable operation up to λ = 1.5.  
- Gasoline produced intermediate IMEP (~13.4 bar at λ = 0.8) with a near-linear lean drop-off.  
- Ammonia yielded the lowest IMEP (~13.2 bar at λ = 0.8) due to slow kinetics and ignition energy demand.  
- A 10 % H₂–NH₃ blend increased IMEP by ~30 % vs pure NH₃, demonstrating hydrogen-stabilisation effects.  
- IMEP hierarchy correlated directly with flame-speed ranking (H₂ > H₂–NH₃ > Gasoline > NH₃).

**Outputs:**  
`/Figures/demo1_imep_lambda.png` — IMEP vs λ, dashed region = fixed burn duration.  

---

## 🔹 Demo 2 — Machine-Learning Grey-Box Correction

**Objective:**  
Demonstrate how machine learning can improve a physics-based model by correcting  
key calibration parameters to match dyno-measured torque.

**Methodology:**
- Baseline single-zone combustion model trained against experimental dyno data.  
- Multilayer Perceptron (MLP) used to learn residuals and correct:
  - Lambda (λ) offset,  
  - FMEP offset (mechanical loss correlation),  
  - Start-of-Combustion (SOC) timing.  
- Model retrained iteratively for accuracy improvement (v1 → v1.1).

**Results Summary:**
- ML correction reduced torque deviation vs dyno from > 5 % to < 1 %.  
- Improved correlation of predicted vs measured IMEP across RPM sweep.  
- Demonstrated the feasibility of combining physics constraints with data-driven residual learning.  

**Outputs:**  
`/Figures/demo2_torque_deviation.png` — Torque comparison: Baseline vs ML v1/v1.1 vs Dyno.

---

## 🚀 Demo 3 — CFD-Informed Combustion Adjustment (WIP)
- Integration of hydrogen CFD data to import mean temperature (T̄), turbulence kinetic energy (k̄) and length-scale.  
- Adjusts laminar and turbulent flame-speed correlations dynamically within the VETB combustion model.

---

## 🧩 Tools & Languages
- Python 3.11  |  NumPy · Pandas · Matplotlib · Scikit-learn  
- OpenFOAM 11 (CFD data import for Demo 3)  
- Jupyter / VS Code environment  



