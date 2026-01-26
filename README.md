<div align="center">

# 🎯 DPOA: Scenario-Aware Pareto Optimization

**Scenario-Aware Pareto Optimization for Limited-Round Unknown-Opponent Competition**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*PDCAT 2025 | International Conference on Parallel and Distributed Computing, Applications and Technologies*

[📄 Paper](#citation) · [🚀 Quick Start](#quick-start) · [📖 Usage](#usage) · [🔧 Configuration](#configuration)

</div>

---

## 📌 Overview

This project implements a scenario-aware Pareto optimization method for limited-round unknown-opponent competition. The approach constructs a candidate opponent model space and leverages scenario-aware Pareto optimization strategies to train agents, enabling superior performance against unknown opponents.

## 🏗️ Architecture

```
DPOA/
├── get_opp.py          # Step 1: Generate true opponent models
├── get_candidate.py    # Step 2: Generate candidate opponent models
├── expand_model.py     # Step 3: Train agent models
├── condense.py         # Step 4: Condense model space
├── ILVE.py             # Step 5: Run experiments
├── DQN_model.py        # DQN model implementation
├── Mnt.py              # Environment implementation
└── no_sensor/          # No-sensor related modules
    ├── belief_filter.py
    └── ...
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy torch tqdm
```

### Installation

```bash
git clone https://github.com/your-username/DPOA.git
cd DPOA
```

## 📖 Usage

Execute the following steps in order:

### Step 1️⃣ Generate True Opponent Models

Generate true opponent models for testing:

```bash
python get_opp.py
```

Models will be saved to the `single_models/` directory.

### Step 2️⃣ Generate Candidate Opponent Models

Generate a pool of candidate opponent models (possible opponent behavior models):

```bash
python get_candidate.py --num_candidate 10 --itera_num 3 --sub_num 1 --opp_num 1
```

**Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_candidate` | Number of candidate models | 1 |
| `--itera_num` | Number of iterations | 3 |
| `--sub_num` | Number of subject agents | 1 |
| `--opp_num` | Number of opponent agents | 1 |
| `--outset` | Starting model ID | 0 |

Models will be saved to the `candidate_model/` directory.

### Step 3️⃣ Train Agent Models

Train subject agents based on the candidate opponent model set:

```bash
python expand_model.py
```

Configurable parameters in the file:
- `model_j_num`: Number of candidate models to use
- `max_frames`: Maximum training frames
- `explore_rate`: Exploration rate

Models will be saved to the `expand_model/` directory.

### Step 4️⃣ Condense Model Space

Condense the candidate model space using scenario-aware strategies:

```bash
python condense.py
```

This step performs model selection based on scenario similarity and outputs representative model sets at different scales.

### Step 5️⃣ Run Experiments

Execute the complete online learning and evaluation experiment:

```bash
python ILVE.py
```

Supports multiple prediction modes (`predict_mode`):
- `0`: No opponent prediction
- `1`: Multi-model voting prediction (MTA)
- `2`: Bayesian belief filter (POMDP)

## 🔧 Configuration

Main configuration parameters are located in respective script files:

| Script | Parameter | Description |
|--------|-----------|-------------|
| `ILVE.py` | `max_trials` | Maximum number of trials |
| `ILVE.py` | `predict_mode` | Prediction mode (0/1/2) |
| `ILVE.py` | `predict_num` | Number of prediction models |
| `expand_model.py` | `max_frames` | Maximum training frames |
| `condense.py` | `q, w, e` | Target quantities for different compression levels |

## 📊 Output Structure

```
DPOA/
├── single_models/          # True opponent models
│   └── {itera_num}/
│       ├── sub_model_term.pkl
│       └── opp_model_{frames}.pkl
├── candidate_model/        # Candidate opponent models
│   └── {itera_num}/
│       └── id_{m}.pkl
└── expand_model/           # Trained agent models
    └── {model_j_num}/
        └── sub_model.pkl
```

## 📄 Citation

If you use this project in your research, please cite:

```bibtex
@inproceedings{chen2025scenario,
  title={Scenario-Aware Pareto Optimization for Limited-Round Unknown-Opponent Competition},
  author={Chen, Fanke and Mei, JiaKang and Liu, Zongyang and Pan, Yinghui},
  booktitle={International Conference on Parallel and Distributed Computing, Applications and Technologies},
  year={2025},
  organization={Springer}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ by SZU**

</div>
