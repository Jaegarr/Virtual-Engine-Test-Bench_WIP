# 🧪 Virtual Engine Test Bench
*Version: v0.1.0 — Baseline Pipeline Check*

A **Python-based simulation tool** for internal combustion engine (ICE) development.  
This project mimics real-world engine test bench calibration — treating the virtual engine as if it were on a dyno.

It’s designed to start simple (steady‑state torque/power) and grow in fidelity version by version, just like a real test engineer’s workflow.

---

## 🎯 Project Philosophy
The project is structured like a real calibration & test engineering campaign:
- Start with a baseline engine model
- Add control maps and physics incrementally (VE, spark timing, BSFC, turbo, etc.)
- Use real test workflows: RPM sweeps, parameter tuning, transient runs, fault injection
- Log and analyze data to iteratively optimize performance & efficiency

---

## 🧱 Core Components (planned)
✅ Torque & power calculation  
✅ Air mass flow model based on VE  
⬜ Spark timing & knock sensitivity tuning  
⬜ Fuel flow & efficiency (BSFC) modeling  
⬜ Boost & turbo dynamics  
⬜ PID control loops (idle, boost)  
⬜ Fault injection & emissions estimation  
⬜ CLI & later GUI / dashboard

---

## 🚦 Roadmap & Version Milestones

| Version | Name                             | Key Deliverables |
|--------|----------------------------------|------------------|
| v0.1.0 | Baseline Pipeline Check          | Airflow calc, torque & power functions, RPM sweep |
| v0.5.0 | VE Map Integration               | VE vs RPM map; validate airflow curve |
| v1.0.0 | Spark Timing Calibration         | Spark map; torque correction; peak torque sweep |
| v1.1.0 | Calibration Enhancements         | Idle PID, EGR & VVT maps, config save/load |
| v1.5.0 | Combustion & Optimization        | Wiebe function, in-cylinder pressure, IMEP calc, parameter sweep |
| v2.0.0 | Comprehensive Model              | Multi‑fuel support, HIL/ECU stub, emissions cycle |
| v2.5.0 | Turbo & Knock-Limit Layer        | Boost map, knock-limited torque |
| v3.0.0 | Advanced Calibration Suite       | Auto‑tuning (DOE), real-time dashboard, reports |

---

## 💡 Future Extensions
- Variable valve timing & lift modeling
- Hybrid powertrains
- Exhaust aftertreatment simulation
- Thermal modeling (coolant, oil temps)

---

## 🧠 Learning Goals
- Understand real calibration & test engineering workflows
- Combine physics-based models & data-driven tuning
- Build something modular, reusable & industry‑relevant

---

## 📌 Status
> **Currently building:** v0.1.0 — baseline CLI torque & power calculator  
> **Next planned milestone:** VE map integration (v0.5.0)

---

## 📎 Tools & Credits
- Python 3.11+
- Using: `numpy`, `pandas` (later: `matplotlib`, `scipy`, `plotly`, `streamlit`)
- Inspired by real dyno workflows & calibration strategies
