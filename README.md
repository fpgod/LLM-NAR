# 🚀 LLM-NAR 

Welcome to **LLM-NAR**!  
This repository implements our paper:  
**"Neural Algorithmic Reasoners informed Large Language Model for Multi-Agent Path Finding (LLM-NAR)"**.  
This project enhances Large Language Models (LLMs) with GNN-based Neural Algorithmic Reasoners for improved performance on Multi-Agent Path Finding (MAPF) problems.

> 🧠🤖 **LLM-NAR** combines the reasoning ability of LLMs with the algorithmic power of neural models, targeting robust, efficient MAPF solutions for complex environments.


---

**📢 News:**  
Our paper has been **accepted by IJCNN 2025 (International Joint Conference on Neural Networks)** and will be available soon!

If you cannot view ijcnn_video, please check [ijcnn_video2](https://github.com/fpgod/LLM-NAR/blob/main/ijcnn_video2.mp4).

---

## 📦 Installation

We recommend using [conda](https://docs.conda.io/en/latest/) for dependency management.

1. **Clone this repo.**

   ```shell
   $ git clone https://github.com/fpgod/LLM-NAR.git
   $ cd LLM-NAR/code
   ```

2. **Install dependencies.**

   ```shell
   $ conda create -n llm_nar python=3.9
   $ conda activate llm_nar
   $ pip install -r requirements.txt
   ```


---

## 🏗️ Models

The `models/` directory contains:

- Pretrained LLM-NAR model checkpoints

  

---

## 📂 Dataset Structure

```
LLMNARdataset/
├── train/                  # Training dataset (805 instances)
├── gemma2/                 # Solutions by Gemma-2 (70 instances)
├── gpt35/                  # Solutions by GPT-3.5 (83 instances)
├── llama3/                 # Solutions by LLaMA-3 (70 instances)
├── qwen2/                  # Solutions by Qwen-2 (70 instances)
└── LLM-NAR/                # Our LLM-NAR solutions (127 instances)
    ├── starts.npy          # Agent starting positions
    ├── goals.npy           # Agent destination positions
    ├── obstacles.npy       # Obstacle positions
    ├── action_llms.npy     # Action sequences by LLM
    ├── action_gnns.npy     # Action sequences by MAPF-GNN
    ├── action_predicts.npy # Action sequences by LLM-NAR
    └── llm_record.json     # LLM interaction history
```

Each subfolder contains JSON files with:
- Problem instances (grids, agent positions)

- Corresponding solutions

  

---

## 🏃‍♂️ Training

Train the LLM-NAR model with:

```bash
python train.py
```

---

## ✅ Validation

### 1. Standard LLM Evaluation

```bash
python validation_llm.py
```

### 2. LLM-NAR Evaluation

```bash
python validation_llmnar.py
```



---

## 📊 Output Metrics

Both validation scripts generate a `validation_metrics.json` file containing:

| Metric            | Description                                                      |
|-------------------|------------------------------------------------------------------|
| `success_rate`    | Overall success rate across all episodes                         |
| `success_rate%`   | Success rate normalized by the number of agents                  |
| `flow_time`       | Total flow time (sum of agents' arrival times)                   |
| `total_steps`     | Total steps taken by all agents                                  |
| `num_episodes`    | Number of evaluated episodes (problem instances)                 |



---

<p align="center">
  <img src="https://img.icons8.com/color/96/robot-2.png" alt="robot icon" width="64" height="64"/>
  <br>
  <b>Happy Reasoning!</b>
</p>