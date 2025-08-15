# 🧪 Virtual Engine Test Bench
**Version: v0.5.0 — Minimum Viable Product (in progress)**

Python-based simulation tool for internal combustion engine (ICE) development.  
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

## ✅ Current Features (v0.5.0 — Minimum Viable Product)
- **CLI interface** to run single-point or RPM sweep simulations
- **Airflow, torque, power & horsepower** calculated from displacement, RPM, and **VE map interpolation**
- **DataFrame output** with values rounded to 3 decimals, ready for reporting
- **Automatic CSV export** to `/Results`, timestamped & user-named
- **Full-throttle RPM sweep** mode for curve generation  
- **Modular architecture**: simulation (`Test_Modes.py`), reporting (`Reporting.py`), and config separated
- **VE map loader** — user can upload custom VE vs RPM/Load maps or use the built-in default (Nissan 350Z dataset)
- **Simplistic emissions model** estimating CO₂, CO, NOx, THC from fuel flow & AFR (empirical coefficients, user-swappable)
- **FMEP model** applied to improve torque realism
- **Input validation** for VE, RPM, displacement, units
- **Configurable test types** — detect WOT tests automatically or prompt for throttle-position filtering

---

## 🧱 Core Components
| Status | Component |
|--------|-----------|
| ✅ | Baseline torque & power calculator |
| ✅ | Air mass flow model (VE map support) |
| ✅ | CLI simulation modes (single point, RPM sweep) |
| ✅ | CSV export module |
| ✅ | VE map loader & interpolation |
| ✅ | Preliminary emissions model |
| ✅ | FMEP friction model |
| ⬜ | BSFC map integration |
| ⬜ | Spark timing & knock-limited torque |
| ⬜ | Turbocharging model |
| ⬜ | PID control layers (idle, boost) |
| ⬜ | GUI & reporting dashboard |

---

## 🧪 Emissions Model (Experimental)
Current v0.5 emissions estimation includes:
- **CO₂ estimation** from fuel mass flow using carbon ratio
- **HC, CO, NOx, THC estimation** via empirical formulas based on AFR and airflow
- Coefficients easily swappable with real test data

**Planned improvements:**
- BSFC map integration
- Lambda-based emissions profile
- Advanced NOₓ modelling (post-spark integration)

---

## 🚦 Roadmap & Version Milestones
| Version | Name                               | Key Deliverables |
|---------|------------------------------------|---------------------------------------------------------------------|
| v0.1    | Baseline pipeline Check            | CLI driver, torque & power calc, single run & RPM sweep, CSV export |
| v0.5    | Minimum Viable Product             | VE map integration, improved fidelity, basic emissions |
| v1.0    | Empirical Combustion               | Empirical combustion model, fuel flow–based torque refinement |
| v1.1    | Spark Timing                       | Spark-advance mapping, torque correction sweeps |
| v1.2    | Reverse Engine Simulation          | Reverse-calculation mode (torque → airflow), validation tools |
| v1.5    | Transient Testing                  | Step & ramp tests, transient torque response modelling |
| v2.0    | Multifuel & Emissions              | Multi-fuel support, WLTP/NEDC cycle simulation, emissions expansion |
| v2.1    | Automated Reporting                | Auto-plotting, trendlines, PDF/HTML report generation |
| Backlog | —                                  | Turbo model, knock-limit layer, hybrid extensions |

---

## 📊 Planned Validation
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
- Python 3.11+
- `pandas`
- Future: `numpy`, `matplotlib`, `scipy`
- JIRA for project task tracking
- Inspired by real-world ICE development & dyno testing methodology

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
> 🔄 `v0.5.0` in progress — VE map interpolation, validation vs real dyno data, BSFC groundwork, preliminary emissions model  
> ⏭ `v1.0.0` next — empirical combustion modelling & torque refinement
