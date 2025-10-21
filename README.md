# Scaling Laws and Scaffold Effects in Multi-Agent LLM Coordination

<div align="center">

**Understanding How Design Choices Shape Coordination Capabilities in Language Model Agents**

[Paper](#) | [Installation](#installation) | [Quick Start](#quick-start) | [Experiments](#experiments) | [Citation](#citation)

</div>

## Overview

This repository contains the code and analysis for our study on **coordination scaling laws in multi-agent LLM systems**, conducted using the **Collab-Overcooked** benchmark environment. We investigate when and how language models of varying sizes coordinate effectively, and critically, how subtle design choices—scaffolds, prompts, turn order, and environmental structure—dramatically influence their collaborative capabilities.

### Key Findings

Our empirical study reveals four critical insights about LLM agent coordination:

1. **Positive Scaling with Clear Scaffolds**: When agents receive well-defined roles (Chef vs. Assistant in asymmetric settings), coordination follows clean scaling laws—larger models perform better both in self-play and cross-play.

2. **Scaffold-Dependent Scaling**: Removing explicit role definitions (symmetric environment) breaks down scaling regularities. What appears to be intrinsic coordination ability is often an artifact of scaffolding design.

3. **Hierarchy Predicts Success**: The emergence of stable leader-follower structures correlates strongly with task completion. Turn order creates leadership priors, and configurations with larger models as followers (e.g., 14B×32B) often outperform their role-swapped counterparts (32B×14B).

4. **Parallelization Amplifies Coordination**: Tasks with decomposable subtasks (Level 4) enable better work division, shorter trajectories, and stronger scaling signals when agents successfully negotiate roles.

### Why This Matters

As LLMs increasingly operate in multi-agent settings—coordinating with other AI systems and humans in open-ended tasks—understanding the true drivers of coordination is essential. Our work demonstrates that benchmark conclusions about "coordination scaling" can be artifacts of evaluation design. We provide concrete recommendations for more robust multi-agent evaluations.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Experimental Setup](#experimental-setup)
  - [Environments](#environments)
  - [Model Configurations](#model-configurations)
- [Running Experiments](#running-experiments)
- [Analysis and Visualization](#analysis-and-visualization)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Installation

We recommend using conda for environment management. Python 3.9+ is required for compatibility with recent transformer models (Qwen3, Gemma3, etc.).

```bash
# Create and activate conda environment
conda create -n collab-overcooked python=3.9
conda activate collab-overcooked

# Install Python dependencies
pip install -r requirements.txt

# Install MPI (pip installation often fails)
conda install mpi4py==3.1.4

# Install the Overcooked-AI game environment locally
cd ./lib/overcooked_ai
pip install -e .
cd ../..
```
---

## Quick Start

### Testing the Environment

The easiest way to verify your installation is to run a simple test with GPT-3.5-turbo:

```bash
# Add your OpenAI API key
echo "your-api-key-here" > src/openai_key.txt

# Run a 3-timestep test
cd src
python main.py --horizon 3 --order boiled_egg
```

If you see the environmental visualization, agent outputs, and the run completes without errors (~1-2 minutes), your setup is correct.

### Running with Local Models

We recommend using [vLLM](https://github.com/vllm-project/vllm) for efficient local deployment:

```bash
# Example: Running Qwen3.0-8B in asymmetric mode
python main.py \
  --model qwen/Qwen3.0-8B \
  --chef-model qwen/Qwen3.0-14B \
  --assistant-model qwen/Qwen3.0-8B \
  --level 2 \
  --episodes 10
```

---

## Experimental Setup

### Environments

We evaluate agents in two variants of Collab-Overcooked:

#### 1. Asymmetric Environment (Main Experiments)

- **Chef** controls: pot, oven, delivery station
- **Assistant** controls: ingredient dispenser, chopping board, blender, dish stack
- **Shared**: Central counter for handoffs
- **Roles**: Explicitly defined; Chef knows recipes, Assistant prepares ingredients
- **Levels**: 1-5 (increasing complexity)
- **Episodes per pairing**: 250 (5 levels × 5 recipes × 10 episodes)

#### 2. Symmetric Environment (Scaffold Analysis)

- **Physical partition removed**: Both agents access all stations
- **Shared knowledge**: Both agents see the recipe
- **No prescribed roles**: Agents must negotiate task division
- **Levels evaluated**: 2 and 4 (selected for coordination regime diversity)
- **Episodes per pairing**: 100 (2 levels × 5 recipes × 10 episodes)

**Key difference**: Symmetric setting removes role ambiguity from physical constraints but introduces negotiation overhead.

### Model Configurations

#### Primary Lineage (Qwen3.0)
- 1.7B, 4B, 8B, 14B, 32B parameters
- All instruction-tuned variants
- Temperature: 0.7
- Max communication: 4 SAY turns per window
- **Note**: 1.7B excluded from symmetric analysis due to degenerate behavior (see paper Appendix)

#### Cross-Vendor Models
- Gemma 3.0: 4B, 12B
- Llama 3.1: 8B
- NVIDIA Nemotron: 14B

#### Experimental Pairings
- **Self-play**: All combinations of (model_i, model_i)
- **Cross-play**: All ordered pairs (model_i, model_j) where i ≠ j
- **Turn order**: In symmetric mode, we test both (i, j) and (j, i) to measure turn-order effects


---

### Processing Results

**Output files**:
- `statistics_data.csv`: Per-task metrics (RAT similarity, redundancy, collaboration metrics)
- `converted_data.csv`: Per-level aggregated results
- Individual task folders: Detailed logs and visualizations

---

## Analysis and Visualization

All analysis notebooks are in the `analysis/` directory:

### Main Results

```bash
cd analysis

# Asymmetric environment analysis (paper Figures 2-3)
jupyter notebook main-per_level_analysis.ipynb
jupyter notebook main-bar_chart_SEM.ipynb

# Symmetric environment analysis (paper Figures 4-6)
jupyter notebook sym-per_level_analysis.ipynb
jupyter notebook sym-bar_chart_SEM.ipynb

# Cross-vendor analysis
jupyter notebook cross-per_level_analysis.ipynb

# Hierarchy and coordination correlation
jupyter notebook coordination_analysis.ipynb
```

### Key Visualizations Generated

From the paper:
- **Figure 2**: Cross-play success heatmaps (asymmetric, all levels)
- **Figure 3**: Mean RAT similarity by role (Chef vs. Assistant)
- **Figure 4**: Symmetric vs. asymmetric comparison (levels 2 & 4)
- **Figure 5**: Hierarchy classification vs. success rate
- **Figure 6**: Task division balance (action count ratios)

---

## Repository Structure

```
Collab-Overcooked/
├── src/                          # Core experiment code
│   ├── main.py                   # Main experiment runner
│   ├── run_model_combinations.py # Batch experiment script
│   ├── evaluation.py             # Metric computation
│   ├── organize_result.py        # Result aggregation
│   ├── convert_result.py         # Level-wise summaries
│   ├── eval_utils.py             # Evaluation utilities
│   ├── collab/                   # Agent implementations
│   │   ├── agents/               # LLM agent wrappers
│   │   └── prompts/              # Role-specific prompts
│   └── prompts/                  # Scaffold definitions
│       ├── chef_prompt.txt
│       ├── assistant_prompt.txt
│       └── symmetric_prompt.txt
├── lib/overcooked_ai/            # Modified Overcooked environment
│   └── overcooked_ai_py/
│       ├── mdp/                  # Game logic
│       └── data/layouts/         # Level layouts
├── analysis/                     # Jupyter notebooks for analysis
│   ├── main-per_level_analysis.ipynb
│   ├── sym-per_level_analysis.ipynb
│   ├── coordination_analysis.ipynb
│   ├── converted_data_*.csv     # Processed results
│   └── plots_modern/            # Generated figures
├── slurm-scripts/                # Cluster job scripts
├── requirements.txt
└── README.md
```

---

### Pre-computed Results

We provide our complete result CSVs in `analysis/`:
- `converted_data_final-data.csv` (asymmetric, all levels)
- `converted_data_symmetric_data.csv` (symmetric, levels 2 & 4)
- `sym_GPT5_data.csv` (hierarchy classifications)

You can skip expensive experiments and run analysis notebooks directly:

```bash
cd analysis
jupyter notebook coordination_analysis.ipynb
# All paper figures can be regenerated from the CSVs
```

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@inproceedings{meireles2025scaling,
  title={Scaling Laws and Scaffold Effects in Multi-Agent LLM Coordination},
  author={Meireles Mariana, Bhati Rupali, Laufer Niklas, Allen Cameron},
  booktitle={NeurIPS Workshop on Multi-Agent Systems},
  year={2025}
}
```

### Related Work

This project builds on:

```bibtex
@inproceedings{zhang2024proagent,
  title={ProAgent: Building Proactive Cooperative Agents with Large Language Models},
  author={Zhang, Ceyao and Yang, Kaijie and Hu, Siyi and Wang, Zihao and Li, Guanghe and Sun, Yihang and Zhang, Cheng and Zhang, Zhaowei and Liu, Anji and Zhu, Song-Chun and others},
  booktitle={AAAI Conference on Artificial Intelligence},
  volume={38},
  number={16},
  pages={17591--17599},
  year={2024}
}

@inproceedings{carroll2019utility,
  title={On the Utility of Learning About Humans for Human-AI Coordination},
  author={Carroll, Micah and Shah, Rohin and Ho, Mark K and Griffiths, Tom and Seshia, Sanjit and Abbeel, Pieter and Dragan, Anca},
  booktitle={Advances in Neural Information Processing Systems},
  volume={32},
  year={2019}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

We thank:
- The original Collab-Overcooked team for the foundational environment
- CHAI for funding sources and computing resources

---

## Contact

For questions or issues:
- **Open an issue** on this repository
- **Email**: marianameireles@protonmail.com

---

## Contributing

This repository is no longer maintained but I'm happy to help you if you have any questions and want to move this work forward in any way!
