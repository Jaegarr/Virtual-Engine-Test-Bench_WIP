# 🧪 Virtual Engine Test Bench (VETB)

**Version:** v1.0.0 — Empirical Combustion (✅ complete)  
**Next:** v1.1.0 — Spark Timing (🚧 in progress)

Python-based simulation tool for internal combustion engine (ICE) development.  
It emulates a real engine test bench workflow — as if the virtual engine were mounted on a dyno.

---

## 🎯 Project Philosophy
- Start simple: steady-state torque & power from displacement and VE  
- Build iteratively: VE maps, combustion, spark timing, turbo, emissions  
- Mimic test bench workflows: single RPM runs, WOT, RPM sweeps, fault injection  
- Track fidelity improvements: validate against real dyno data over time  
- Prioritize clean design: CLI-first, modular code, later dashboards & automation  

---

## ✅ Current Features (v1.0.0)
- CLI interface with single-point & RPM sweep simulations  
- Physics-based **Wiebe combustion model** (single-zone heat release)  
- Full **thermodynamic solver** from IVC → IVO  
- Computation of **IMEP, BMEP, FMEP, PMEP**  
- **CA10/50/90** burn duration markers  
- Engine database with geometry + VE maps (no constant VE option)  
- Automatic CSV export to `/Results` (timestamped & user-named)  
- p–V diagrams, torque/power/emissions plots with CA markers  
- Empirical emissions model: CO₂, CO, NOx, THC from AFR & fuel flow  
- BSFC and emissions intensities (g/kWh)  
- Input validation & boundary checks  

---

## 🧱 Core Components
| Status | Component |
|--------|-----------|
| ✅ | Baseline torque & power calculator |
| ✅ | Air mass flow model (VE map support) |
| ✅ | CLI simulation modes (single point, RPM sweep) |
| ✅ | CSV export module |
| ✅ | VE map loader & interpolation |
| ✅ | Emissions model (empirical) |
| ✅ | FMEP/PMEP loss models |
| ✅ | Combustion Model – Wiebe (Single Zone) |
| ⬜ | Spark timing & calibration (MBT/knock phasing) |
| ⬜ | Turbocharging model |
| ⬜ | PID control layers (idle, boost) |
| ⬜ | Multifuel Database |
| ⬜ | Automated Test Cycles (WLTP, NEDC) |
| ⬜ | GUI & reporting dashboard |
| ⬜ | Reverse Engine Simulation for performance analysis |

---

## 🚦 Roadmap & Milestones
| Version | Name | Key Deliverables |
|---------|------|------------------|
| v0.1 | Baseline Pipeline | CLI driver, torque & power calc, CSV export |
| v0.5 | Minimum Viable Product | VE map, FMEP/PMEP, basic emissions, plotting |
| v1.0 | Empirical Combustion | Wiebe combustion, p–V loops, IMEP/BMEP |
| v1.1 | Spark Timing | Spark-advance mapping, torque correction sweeps |
| v1.5 | Transient Testing | Step & ramp tests, idle-speed PID, inertial dynamics |
| v2.0 | Multi-fuel & Emissions | Fuel database, aftertreatment, WLTP/NEDC cycles |
| v2.1 | Automated Reporting | Dash/UI dashboards, PDF/HTML report generation |
| v2.5 | Reverse Engine Simulation | OBD-data ingestion, back-calibration validation |

---

## 📊 Validation
- Default VE map: Nissan VQ35DE (3.5L NA, Nissan 350Z)  
- Model validated vs dyno torque & power curves  
- ~2% improvement in accuracy vs v0.5  
- Main discrepancy at 4500 rpm due to VVT/VVL effects not yet modeled  
- Expansion & blowdown behaviour consistent with expected SI combustion  

---

## 💾 Data & Reporting
- Results saved as `.csv` in `/Results`  
- Timestamped filenames avoid overwriting  
- Plotting available for torque, power, p–V loops, and emissions  
- Future: auto-generated reports (PDF/HTML)  

---

## 🧠 Learning Goals
- Learn Python through ICE calibration workflows  
- Develop calibration-engineer mindset (combustion, spark, knock, BSFC)  
- Practice modular CLI-first design patterns  
- Understand emissions formation & estimation strategies  
- Move toward reproducible, shareable simulation tools  

---

## 📎 Tools & Dependencies
- Python 3.13+  
- numpy, pandas, scipy, matplotlib  
- JIRA for project tracking  

---

## 📺 Demo
[WOT Test Demo]([url](https://drive.google.com/file/d/18G-wP4hhR3n0aM7SgCmowkMYLoXPCNX9/view?usp=sharing))

---

### Status
- ✅ v0.1.0 complete — baseline pipeline  
- ✅ v0.5.0 complete — MVP with VE maps, emissions, plotting  
- ✅ v1.0.0 complete — combustion model (Wiebe function, p–V loops)  
- 🚧 v1.1.0 in progress — spark timing & calibration  
