# Emergency Evacuation Planning via Dynamic Network Flow

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![React](https://img.shields.io/badge/React-Vite-61DAFB)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Theory-red)

## 📌 Overview

This project addresses the **Emergency Evacuation Planning** problem using techniques from network flow theory. The objective is to determine the **minimum time horizon ($T^*$)** required to evacuate all individuals from a building to safety, subject to hallway capacities and transit times.

Theoretically, this models the **Quickest Transshipment Problem**. We solve it by reducing the dynamic flow problem over time into a sequence of static **Maximum Flow** problems using **Time-Expanded Graphs**.

## 🚀 Key Features

* **Time-Expanded Graph Construction:** Converts dynamic building flows into a static layered graph.
* **Binary Search Optimization:** Efficiently finds the minimal evacuation time $T^*$ without guessing.
* **Bottleneck Analysis:** Identifies capacity-constrained vs. transit-constrained scenarios.
* **Interactive Visualization:** A React-based web interface to simulate and visualize evacuation dynamics.
* **Scalability Testing:** Verified performance on multi-floor building topologies.

---

## 📂 Project Structure

```text
evacuation-project/
├── algorithm/                  # Python Implementation
│   ├── emergency_evacuation_planning.py  # Main algorithm & simulations
│   ├── experiment1_density.csv           # Results data
│   └── experiment3_scalability.csv       # Results data
├── visualization/              # Interactive React Web App
│   ├── src/
│   │   └── components/
│   │       └── BuildingEvacuationViz.jsx
│   ├── App.jsx
│   ├── main.jsx
│   └── package.json
├── references.bib              # Bibliography for the report
└── README.md                   # This file
