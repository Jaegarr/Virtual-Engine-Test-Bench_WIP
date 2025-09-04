# 🧪 Virtual Engine Test Bench (VETB)

**Version:** v0.5.0 — Minimum Viable Product (✅ complete)  
**Next:** v1.0.0 — Empirical Combustion (🚧 in progress)

Python-based simulation tool for internal combustion engine (ICE) development.  
It emulates a real engine test bench workflow — as if the virtual engine were mounted on a dyno.

---

## 🎯 Project Philosophy
- Start simple: steady-state torque & power from displacement and VE  
- Build iteratively: VE maps, fuel flow, spark timing, turbo, emissions  
- Mimic test bench workflows: single RPM runs, WOT, RPM sweeps, fault injection  
- Track fidelity improvements: validate against real dyno data over time  
- Prioritize clean design: CLI-first, modular code, later dashboards & automation  

---

## ✅ Current Features (v0.5.0)
- CLI interface with single-point & RPM sweep simulations  
- Airflow, torque, power & horsepower from displacement, RPM, VE map interpolation  
- Automatic CSV export to `/Results` (timestamped & user-named)  
- VE map loader (custom map or default Nissan VQ35DE dataset)  
- Empirical emissions model: CO₂, CO, NOx, THC from AFR & fuel flow  
- FMEP/PMEP models for internal losses  
- WOT & full-sweep plotting (torque, power, emissions curves)  
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
| ✅ | Preliminary emissions model |
| ✅ | FMEP/PMEP loss models |
| ⬜ | Combustion Model - Wiebe (Single Zone) |
| ⬜ | Spark timing & calibration |
| ⬜ | Turbocharging model |
| ⬜ | PID control layers (idle, boost) |
| ⬜ | Multifuel Database |
| ⬜ | Automated Test Cycles (WLTP, NEDC)|
| ⬜ | GUI & reporting dashboard |
| ⬜ | Reverse Engine Simulation for performance analysis |

---

## 🚦 Roadmap & Milestones
| Version | Name | Key Deliverables |
|---------|------|------------------|
| v0.1 | Baseline Pipeline | CLI driver, torque & power calc, CSV export |
| v0.5 | Minimum Viable Product | VE map, FMEP/PMEP, basic emissions, plotting |
| v1.0 | Empirical Combustion | Wiebe-function combustion, torque refinement |
| v1.1 | Spark Timing | Spark-advance mapping, torque correction sweeps |
| v1.5 | Transient Testing | Step & ramp tests, idle-speed PID, inertial dynamics |
| v2.0 | Multi-fuel & Emissions | Fuel database, aftertreatment, WLTP/NEDC cycles |
| v2.1 | Automated Reporting | Dash/UI dashboards, PDF/HTML report generation |
| v2.5 | Reverse Engine Simulation | OBD-data ingestion, back-calibration validation |

---

## 📊 Validation
- Default VE map: Nissan VQ35DE (3.5L NA, Nissan 350Z)  
- Model validated vs dyno torque & power curves  
- Agreement in mid-range, overprediction at low/high RPM due to constant efficiency & friction model limits  
- Continuous improvements planned (combustion, spark, transient dynamics)  

---

## 💾 Data & Reporting
- Results saved as `.csv` in `/Results`  
- Timestamped filenames avoid overwriting  
- Plotting available for torque, power & emissions  
- Future: auto-generated reports (PDF/HTML)  

---

## 🧠 Learning Goals
- Learn Python through ICE calibration workflows  
- Develop calibration-engineer mindset (VE, AFR, spark, knock, BSFC)  
- Practice modular CLI-first design patterns  
- Understand emissions formation & estimation strategies  
- Move toward reproducible, shareable simulation tools  

---

## 📎 Tools & Dependencies
- Python 3.13+  
- numpy, pandas, scipy,  matplotlib  
- JIRA for task tracking  

---

## 📺 Demo
[WOT Test Demo](https://drive.google.com/file/d/1-dpdAOZIZWzUSkz9k_nm_YxfedbN-AH8/view?usp=sharing)

---

### Status
- ✅ v0.1.0 complete — baseline pipeline  
- ✅ v0.5.0 complete — MVP with VE maps, emissions, plotting  
- 🚧 v1.0.0 in progress — combustion model (Wiebe function)  
