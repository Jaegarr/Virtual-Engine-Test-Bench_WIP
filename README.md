# 🧪 Virtual Engine Test Bench
**Version: v0.1.0 — Baseline Pipeline Complete; now advancing toward v0.5.0**

A **Python-based simulation tool** for internal combustion engine (ICE) development.  
It emulates a real engine test bench workflow — as if the virtual engine were mounted on a dyno.

The project begins with a simplified torque & power model and evolves in fidelity with each version — reflecting how a calibration engineer incrementally builds model complexity.

---

## 🎯 Project Philosophy
- **Start simple**: steady-state torque & power from displacement and VE
- **Build iteratively**: layer in VE maps, fuel flow, spark timing, knock, turbo, emissions
- **Mimic test bench workflows**: single point runs, RPM sweeps, fault injection
- **Track fidelity improvements**: validate against real-world dyno data over time
- **Prioritize clean design**: CLI-first, modular code, later dashboards & automation

---

## ✅ Current Features (v0.1.0)
- 🔧 **CLI interface** to run single-point or RPM sweep simulations
- ⚙️ **Airflow, torque, power & horsepower** calculated from displacement & VE
- 📄 **Pretty DataFrame output** with values rounded to 3 decimals
- 📦 **Automatic CSV export** to `/Results`, timestamped & user-named
- 📈 **Basic RPM sweep mode** for full-throttle curve generation
- 🧱 **Modular architecture**: separate simulation (`Test_Modes.py`) and reporting (`Reporting.py`)
- ✅ **Input validation** for VE, RPM, displacement, units

---

## 🧱 Core Components
✅ Baseline torque & power calculator  
✅ Air mass flow model (constant VE)  
✅ CLI simulation modes (single point, RPM sweep)  
✅ CSV export module  
✅ VE map loader (ready for v0.5.0)  
✅ Preliminary emissions model (fuel-based estimation)  
⬜ BSFC map integration  
⬜ Spark timing, knock-limited torque  
⬜ Turbocharging model  
⬜ PID control layers (idle, boost)  
⬜ GUI & reporting dashboard

---

## 🧪 Emissions Model (Experimental)
Version 0.1.1 adds a **simplistic emissions model** using fuel flow and AFR.  
Current capabilities include:
- **CO₂ estimation** (from fuel mass flow using carbon ratio)
- **HC and CO estimation** (empirical formulas based on AFR and airflow)
- Empirical coefficients easily swappable with real test data

Planned improvements:
- BSFC map integration
- Lambda-based emissions profile
- NOₓ estimation (post-spark integration)

---

## 🚦 Roadmap & Version Milestones

| Version | Name                          | Key Deliverables |
|--------|-------------------------------|------------------|
| v0.1.0 | Baseline Pipeline Check       | CLI driver, torque & power calc, single run & RPM sweep, CSV export |
| v0.1.1 | Emissions Estimator           | Fuel flow model, CO₂, HC, CO estimation |
| v0.5.0 | VE Map Integration            | Load VE vs RPM map from CSV, improve fidelity & validate vs real data |
| v1.0.0 | Spark Timing Calibration      | Spark-advance map, torque correction sweeps |
| v1.1.0 | Calibration Enhancements      | Idle PID, VVT maps, EGR maps |
| v1.5.0 | Combustion & Optimization     | Wiebe function, in-cylinder pressure, IMEP calc |
| v2.0.0 | Comprehensive Model           | Multi-fuel support, emissions cycle simulation (WLTP, NEDC) |
| v2.5.0 | Turbo & Knock-Limit Layer     | Boost pressure, knock-limited torque |
| v3.0.0 | Advanced Calibration Suite    | Auto-tuning (DOE), transient tests, dashboard & reporting

---

## 📊 Planned Validation
To demonstrate model growth and accuracy:
- Compare model vs real SI engine dyno data:
  - Torque & power curves
  - VE maps
  - Spark & knock-limited torque
  - Emissions vs logged cycle data (if available)
- Visualize improvements version by version

---

## 💾 Data & Reporting
- Exports `.csv` files via the `Reporting` module
- Timestamped filenames to avoid overwrite
- Default save path: `/Results` (auto-created)
- Future: auto-generated plots, trendlines, and full PDF/HTML reports

---

## 🧠 Learning Goals
- Learn Python through a project that mirrors real-world powertrain workflows
- Build calibration-engineer thinking (VE, AFR, spark, knock, BSFC)
- Practice modular, CLI-first design patterns
- Understand emissions formation & estimation strategies
- Move toward reproducible, shareable engine simulation tools

---

## 📎 Tools & Dependencies
- Python 3.11+
- `pandas` (used)
- Future: `numpy`, `matplotlib`, `scipy`
- JIRA for project task tracking
- Project inspired by real-world ICE development & dyno testing methodology

---

## 💡 Future Extensions
- Hybrid & electrified powertrains
- Exhaust aftertreatment & thermal modeling
- Knock prediction & octane sensitivity
- Auto-tuned calibration (DoE-style)
- Live dashboards with data visualizations

---

> **Status:**  
> ✅ `v0.1.0` complete — baseline CLI + airflow + torque/power  
> 🔜 `v0.5.0` next — VE map interpolation, validation, BSFC groundwork, emissions  
