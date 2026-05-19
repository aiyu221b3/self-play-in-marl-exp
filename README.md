# Emergent Market Behavior through MARL 📈

> 🚧 **Work in Progress:** This repository contains the preliminary code and visualization tools for the upcoming survey paper: *Emergent market behavior through MARL: self-play and decentralized learning in financial systems*. The repository is actively being structured for official release.

This repository implements a lightweight, Tabular Q-Learning environment designed to isolate, observe, and mathematically prove the core challenges and emergent phenomena of Multi-Agent Reinforcement Learning (MARL) within simulated financial markets.

## 🔬 Core Experiments

The codebase is split into two foundational environments, avoiding deep neural architectures to maintain mathematical clarity and focus purely on Game Theory dynamics.

### 1. The Zero-Sum Baseline (1v1)
A pursuit-evasion game mapping to **Imperfect Information** market conditions.
* **Informed Trader:** Attempts to reach the target accumulation zone while minimizing distance penalties.
* **Market Maker:** Attempts to intercept the trader with a limited vision radius (Partial Observability).
* **Key Findings:** Demonstrates non-stationarity, reward oscillation, and emergent deception (the trader learning to skew action distributions toward "Stay" to mask intent).

### 2. The Mixed-Motive Bottleneck (2v1)
A complex environment introducing **Spurious Coordination** and the **Free-Rider Problem** without programmed teamwork.
* **Fast Trader ($\alpha = 0.5$):** Adapts rapidly, learning to act as an intentional decoy.
* **Slow Trader ($\alpha = 0.05$):** Acts as market noise, stumbling into the target zone while the Market Maker is distracted.
* **Key Findings:** Proves how local, self-interested reward updates can breed global, cooperative market manipulation. 

## 📊 Visualizations

The training scripts automatically generate publication-ready, high-resolution PDFs tracking:
* Non-Stationarity & Rolling Win Rates
* Spatial Occupancy Heatmaps (Agent Footprints)
* Action Distribution Evolution (Emergent Deception)
* Episode Efficiency / Duration Convergence
* Self-Play Evaluation Curves

## 🚀 Quick Start

To run the baseline 1v1 zero-sum training loop and generate the kinematic metrics:

```bash
# Clone the repository
git clone https://github.com/aiyu221b3/self-play-in-marl-exp.git
cd marl-market-behavior

# Install dependencies
pip install numpy matplotlib pandas

# Execute the 1v1 baseline
python baseline_1v1.py

# Execute the 2v1 mixed-motive environment
python multi_agent_2v1.py
📝 Citation
If you use this environment or the resulting visualizations in your research, please cite the author:

Code snippet
@article{ayushi2026marl,
  title={Emergent market behavior through MARL: self-play and decentralized learning in financial systems},
  author={Ayushi Bhattacharya},
  year={2026},
  publisher={GitHub}
}
```

## 🛠️ Acknowledgments
* The core MARL environment and Game Theory logic were developed independently, and through criticism from **Claude**.
* **Google Gemini** was utilized to assist with refactoring the Matplotlib scripts to achieve publication-ready visualization aesthetics.
