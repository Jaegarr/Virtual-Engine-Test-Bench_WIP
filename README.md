# 🧪 Virtual Engine Test Bench

A Python-based simulation tool for internal combustion engine (ICE) development.  
This project mimics the real-world process of engine test bench calibration, treating the virtual engine as if it were on a dyno.

It allows performance analysis, fuel efficiency evaluation, and map-based parameter tuning—starting simple and growing in fidelity version by version.

---

## 🎯 Project Philosophy

Instead of just building an engine simulator, this project is treated like a real **test engineering campaign**:

- Start with a first-run baseline engine
- Add sensor and control maps incrementally (VE, spark timing, BSFC, boost, etc.)
- Use real test engineer workflows: RPM sweeps, parameter tuning, fault injection
- Evaluate performance, log results, and optimize the engine via simulation

---

## 🧱 Core Components (to be developed)

- **Air Mass Flow Model** based on VE and intake conditions  
- **Torque & Power Calculation** from fuel and spark inputs  
- **Fuel Flow and Efficiency Models** using BSFC maps  
- **Spark Timing and Knock Sensitivity** tuning  
- **Boost & Turbo Modeling** (for later versions)  
- **Interactive Interface** for map tuning and test control  
- **Fault Injection and PID Control** modules

---

## 🚦 Roadmap & Version Milestones

| Version | Name                             | Key Deliverables                                                                                                     |
|---------|----------------------------------|----------------------------------------------------------------------------------------------------------------------|
| **v0.1**  | Baseline Pipeline Check         | Constants for VE, spark, BSFC; CLI RPM sweep; torque & power plot                                                   |
| **v0.5**  | VE Map Integration              | VE vs RPM map; air mass flow vs RPM; validated shape & behavior                                                     |
| **v1.0**  | Spark Timing Calibration        | Spark-advance map; torque correction factor; sweep for peak torque vs timing                                        |
| **v1.5**  | BSFC & Efficiency Mapping       | BSFC map; fuel-flow calculation; BSFC contour plots; CSV logging                                                    |
| **v2.0**  | Optimization Loop & DOE         | Objective-based tuning (e.g. max torque); grid/heuristic search; auto run logging                                   |
| **v2.5**  | Turbo & Knock-Limit Layer       | Boost pressure map; adjusted VE; knock-limit enforcement; knock-limited torque plots                                |
| **v3.0**  | Control & Fault-Injection Suite | PID throttle/boost control; misfire/sensor fault injection; emissions estimation                                    |
| **v3.x**  | Polish, Docs & Demo Package     | Full documentation; project tutorial; install scripts; video demo or GIFs                                           |

---

## 💡 Future Extensions

- Variable valve timing modeling
- Hybrid engine modes (ICE + electric motor)
- Exhaust aftertreatment simulation
- Engine thermal modeling (coolant temp, oil temp)

---

## 🧠 Learning Goals

This project is part of an ongoing journey to:
- Learn Python through applied engineering problems
- Simulate engine behavior from first principles
- Understand calibration logic and test methodologies
- Bridge the gap between data, physics, and performance

---

## 📌 Status

Currently working on: **v0.1 – Baseline pipeline**  
Estimated next milestone: **VE map integration (v0.5)**

---

## 📎 Credits & Tools

- Python 3.11+
- Libraries: `numpy`, `matplotlib`, `pandas` (later: `scipy`, `plotly`, `streamlit`)
- Inspired by real dyno testing workflows

---
