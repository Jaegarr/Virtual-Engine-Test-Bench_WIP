# 🧪 Virtual Engine Test Bench
*Version: v0.1.0 — Baseline Pipeline Complete; moving to v0.5.0*

A **Python-based simulation tool** for internal combustion engine (ICE) development.  
It emulates a real engine test bench workflow as if the virtual engine were mounted on a dyno.

The project starts with a basic torque & power model, and grows in fidelity version by version — mirroring how a calibration engineer incrementally refines an engine model.

---

## 🎯 Project Philosophy
- Start simple: steady-state torque & power from displacement and VE
- Add complexity step by step: VE maps, spark timing, knock, turbo, transient behavior
- Use test-bench style workflows: single point, RPM sweep, full-throttle, fault injection
- Validate against real engine data to track accuracy improvements
- Build modular, maintainable, CLI-first code — later add dashboards & automated reports

---

## ✅ Current Features (v0.1.0)
- **Python CLI driver script** with mode select: single run, RPM sweep, or exit
- **Air mass flow & torque calculation** from displacement & VE
- **Power & horsepower calculation**
- **Pretty-printed pandas DataFrame** (results rounded to 3 decimals)
- **CSV export** via `Reporting` module — automatic timestamp & user-defined filename/folder
- **Basic full-throttle RPM sweep** test mode
- Defensive input validation for RPM, VE, displacement
- Modular design: `Test_Modes.py` (simulation logic), `Reporting.py` (data export)
- Default export location: `/Results` folder created automatically inside the project

---

## 🧱 Core Components
✅ Baseline torque & power calculator  
✅ Air mass flow model (constant VE)  
✅ Horsepower calculation  
✅ CLI interface & mode select  
✅ CSV export & file/folder naming logic  
✅ VE map (MAP, RPM vs VE) from real data  
⬜ Spark timing & knock-limited torque  
⬜ Fuel flow, BSFC & CO₂ estimation  
⬜ Boost & turbocharger modeling  
⬜ PID controllers (idle, boost control)  
⬜ Emissions & aftertreatment estimation  
⬜ GUI / dashboard & automated plots

---

## 🚦 Roadmap & Version Milestones

| Version | Name                          | Key Deliverables |
|--------|-------------------------------|------------------|
| v0.1.0 | Baseline Pipeline Check       | CLI driver, torque & power calc, single run & RPM sweep, CSV export |
| v0.5.0 | VE Map Integration            | Load VE vs RPM map from CSV, improve fidelity & validate vs real data |
| v1.0.0 | Spark Timing Calibration      | Spark-advance map, torque correction sweeps |
| v1.1.0 | Calibration Enhancements      | Idle PID, VVT maps, EGR maps |
| v1.5.0 | Combustion & Optimization     | Wiebe function, in-cylinder pressure, IMEP calc |
| v2.0.0 | Comprehensive Model           | Multi-fuel support, emissions cycle simulation (WLTP, NEDC) |
| v2.5.0 | Turbo & Knock-Limit Layer     | Boost pressure, knock-limited torque |
| v3.0.0 | Advanced Calibration Suite    | Auto-tuning (DOE), transient tests, dashboard & reporting

---

## 📊 Planned Validation
To demonstrate improvement:
- Validate model vs real SI engine data:
  - Torque & power curves
  - VE maps
  - Spark timing and knock data
  - Emissions if available
- Visualize differences & improvements version by version

---

## 💾 Data & Reporting
- Export results as `.csv` via `Reporting` module
- Automatic timestamped filenames to avoid overwriting
- Default folder: `/Results` inside project (auto-created)
- Future: automatic matplotlib plots, PDF/HTML reports

---

## 📌 Current Status
> ✅ v0.1.0 done: baseline torque/power model, single & sweep modes, data export  
> 🛠 Next: v0.5.0 — load VE map, validate against real engine data

---

## 🧠 Learning Goals
- Learn Python through a real-world simulation project
- Apply calibration & test engineering workflows
- Understand how VE, spark, and turbo maps interact in powertrain models
- Practice modular design & defensive coding
- Create reproducible, industry-relevant simulation tools

---

## 📎 Tools & Credits
- Python 3.11+
- `pandas` (later: `numpy`, `matplotlib`, `scipy`)
- JIRA for project management
- Based on real dyno calibration workflows
- Inspired by automotive engine testing practices

---

## 💡 Future Extensions
- Variable valve timing & lift
- Hybrid & electrified powertrains
- Exhaust aftertreatment & thermal modeling
- Knock sensitivity & octane rating
- Automatic reporting dashboards
