# 😄 SMILE: Small Language Model Integrated LIKE Engine

---

## 🚀 Why accelerate the **LIKE** predicate with language models?

Modern databases often use the `LIKE` predicate to search text data. However, when the search condition is broken up by wildcards (`%`, `_`), existing search structures can degrade to the worst-case linear scan 🐌, leading to poor performance. Traditional methods like B+-trees 🌳 struggle when wildcards appear on both ends of the pattern.

Recent advances in the language-model world 💡 open up a very promising new path. These models can “understand” and **decode** complex `LIKE` patterns, converting them into a small set of candidate values 🤖🔍, which can then be verified via hash-table lookups in constant time ⚡—dramatically improving efficiency! But… integrating an LLM directly into a database is **difficult** due to high latency ⏱️, large storage overhead 🗄️, and sensitivity to data drift 🎯.

---

## 🤖 Meet SMILE: Small Language Model Integrated LIKE Engine 😄

![SMILE](./SMILE.png)

SMILE learns **column-local character distributions** using small yet refined parameters ✨. It acts as a lightweight “neural translator” 🔄, translating `LIKE` patterns into a candidate set—fast 🏎️, accurate 🎯, and lightweight 🪶.

### 🏆 Why SMILE?

We evaluate SMILE on multiple datasets (e.g., TPC-H, IMDB, Reddit) and compare it against
PostgreSQL native indexes (B-tree, GIN, GiST) as well as large language models (e.g., DeepSeek-V3, Qwen2.5).

* 🚀 **Stunning speed**: up to **1000×** faster than sequential scan (SeqScan), and **1.8–41.6×** faster than
  PostgreSQL **GIN (Trigram)** index.
* 🎯 **High recall**: under complex query patterns, LLM-based approaches often achieve **<10%** recall,
  while **SMILE consistently reaches 90%–95%+**.
* 🗄️ **Huge storage savings**: compared to **GIN** and **GiST** indexes, SMILE reduces space usage by **23–82×**.


---

## 🗂️ Code Structure

```text
.
├── data                           📁 Demo datasets
│   ├── lineitem.csv               🧾 Sampled TPCH-lineitem data
│   └── wiki.csv                   📚 Wiki text data
├── E2E_Exp                        ⚙️  End-to-end experiment scripts (PostgreSQL)
│   ├── create_index.py            🛠️  Create index / export data
│   ├── generateworkload.py        🎲  Generate LIKE query workloads
│   └── run_workload.py            🚀  Run workloads and evaluate
├── models                         🤖 Pretrained model parameters
│   ├── lineitem                   📦 lineitem model
│   │   └── best_model.pth         ✅ Pretrained weights
│   └── wiki                       📦 wiki model
│       └── best_model.pth         ✅ Pretrained weights
├── SLM_LIKE.py                    🏋️  Model training entry
├── evaluate.py                    🏁 Model evaluation entry
├── chat_inference.py              💬 Interactive inference entry
├── requirements.txt               📦 Python dependencies
└── Realworld_84047LIKE.csv        🌍 84,047 real-world LIKE scenarios
```

---

## 📦 Dependency Requirements

See `requirements.txt` for the full dependency list.

---

## 🛠️ Environment Setup

### 1) Create an environment

```bash
conda create -n SMILE python=3.9 -y
conda activate SMILE
```

### 2) Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📁 Datasets

Data available directly in the repository:

* `./data/lineitem.csv`
* `./data/wiki.csv`

We also added **84,047 real-world LIKE predicate usage scenarios from 30 datasets** in `Realworld_84047LIKE.csv`.

### Open datasets used in the paper

* **IMDB** (primaryName, ~15M): [https://datasets.imdbws.com/](https://datasets.imdbws.com/)
* **WIKI** (titles, ~4M): [https://dumps.wikimedia.org/enwiki/](https://dumps.wikimedia.org/enwiki/)
* **TPC-H** (lineitem.comment, ~24M): [https://www.tpc.org/tpch/](https://www.tpc.org/tpch/)
* **Reddit** (usernames, ~2M): [https://www.kaggle.com/datasets/colinmorris/reddit-usernames](https://www.kaggle.com/datasets/colinmorris/reddit-usernames)
* **RedPajama** (text windows, ~1B): [https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt](https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt)
* **Newsroom** (news articles, ~1.21M): [https://lil.nlp.cornell.edu/newsroom/download/index.html](https://lil.nlp.cornell.edu/newsroom/download/index.html)

---

## 📈 Evaluate SMILE

### Evaluate with the pretrained lineitem model

```bash
python evaluate.py --model_path ./models/lineitem/best_model.pth
```

### Evaluate with the pretrained wiki model

```bash
python evaluate.py --data_path ./data/wiki.csv --model_path ./models/wiki/best_model.pth
```

---

## 🧪 Train SMILE

You can train SMILE yourself on an affordable 8GB GPU and easily verify its performance. It’s very convenient and cost-effective—you can get started right away. Once you plug in the model, you’ll see the magic immediately! 🌟

```bash
python SLM_LIKE.py
```

💡 *Tip: Adjust `--inPct` and `--pct` in the code to control the proportion of queries included and the wildcard ratio.*

After training, the model will be saved to:

* `./models/<saveName>/...` (depending on the script’s actual saving logic)

---

## 💬 Interactive LIKE Pattern Prediction (Interactive Inference) 🤖✨

We provide an **interactive program** 🕹️ that lets you enter an SQL `LIKE` pattern 🔍 (e.g., `%dam La_berth%`, `%mit Sur_avanshi%`, `%ichael _empson%`, `%lexander _ohnson%`, `%aia R_ssell%`), and our SMILE model 😄⚡ will produce **instant predictions** as matching results.

Just type your pattern and press Enter ⌨️—our lightweight neural engine will return predicted matches in real time 🎯!

You can exit anytime by entering `'exit'` or `'q'` ❌👋.

### ▶️ How to run (lineitem)

```bash
python chat_inference.py --data_path ./data/lineitem.csv --model_path ./models/lineitem/best_model.pth --inferSampleNum 4
```

### ✅ Inference example (real runtime output)

Below is an example of a real interactive inference session: you input a complex pattern containing `%` and `_`, and the model returns the Top-4 candidate results and latency.

```text
Model loaded. Enter a LIKE pattern (e.g., a%cd_). Type 'exit' or 'q' to quit.

LIKE pattern > %aia R_ssell%


Results (Top 4):
1. Maia Rossell
2. Maia Rossell
3. Maia Rossell
4. Maia Rossell
Time: 0.16 s

LIKE pattern > exit
Exiting.
```

### ▶️ How to run (wiki, optional)

```bash
python chat_inference.py --data_path ./data/wiki.csv --model_path ./models/wiki/best_model.pth --inferSampleNum 4
```

---

### 🧙 What it does:

* 🗣️ **Chat with SMILE**: easily input `LIKE` patterns in a natural way.
* ⚡ **Instant results**: get predictions in the blink of an eye.
* 🎯 **Real-world queries**: test on real entries from the `lineitem` dataset.
* 💡 **Smart matching**: handles wildcards `%` and `_` in an intelligent learned way.
* 📊 **Neural LIKE acceleration**: simulates queries in real scenarios (e.g., search engines) such as
  `SELECT * FROM lineitem WHERE comment LIKE "%keyword%" LIMIT K`.

---

> 🤖 Tip: `%` matches any number of characters, and `_` matches exactly one character.

> 🧩 *Behind the scenes*: your pattern is sent to our small-but-powerful SMILE model 🤖, which predicts the set of rows that match your pattern like a translator—**much faster** than scanning the entire column linearly. 🔥

---

## 🧱 End-to-end scripts (E2E_Exp)

`E2E_Exp/` includes:

* `create_index.py`: requires connecting to PostgreSQL, and uses `--csv_filename` to specify the output file
* `generateworkload.py`: query generator
* `run_workload.py`: executes evaluation

Notes:

* The input/output formats and default paths of these scripts may need to be adjusted to fit your experimental environment before running.

Whether you’re debugging, testing queries, or just curious—this workflow lets you explore SMILE in a fun and interactive way! 🤓🎉

Ready to chat with your database? 💬📊 Let the LIKE magic begin! ✨🪄

Ready to make your database **smile**? 😄
Let the neural LIKE acceleration begin! ⚡🤖📚