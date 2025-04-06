------

# 🚀 **The Case For Language Model Accelerated** **LIKE** **Predicate**

Modern databases often use the `LIKE` predicate to search text data. However, when the search condition is interrupted by wildcards (`%`, `_`), the existing search structure can degrade into a worst-case linear scan 🐌, resulting in poor performance. Traditional methods like B+-trees 🌳 struggle when wildcards appear at **both ends** of the pattern.

Recent advances in language models 💡 open a promising new path. These models can "understand" and **decode** complex `LIKE` patterns into a small set of candidate values 🧠🔍, which are then verified in constant time using hash table lookups ⚡—greatly improving efficiency! But… integrating LLMs directly into databases is **hard** due to high latency ⏱️, large storage 🗄️, and sensitivity to data drift 🎯.

------

### 🧠 Meet **SMILE**: *Small language Model Integrated LIKE Engine* 😄

SMILE learns **column-local character distributions** through small but exquisite parameters ✨. It acts as a *neural translator* 🔄 that converts `LIKE` patterns into their corresponding result sets—fast 🏎️, precise 🎯, and lightweight 🪶.

### 🏆 Why SMILE?

- 🚀 Up to **1000x faster** than sequential scans & large LLMs
- 🔥 **44–141x faster** than trigram indexes
- ⚡ **100x faster** than B+-trees
- 🤖 Only a **tiny model** (5 orders of magnitude smaller than GPT-scale models!)
- 💪 Robust against data/query distribution drifts

------

## 🗂️ Code Structure

Our project is organized as follows:

```
.
├── data                           📁 Demo dataset (Sampled TPCH-lineitem)
├── E2E_Exp                        ⚙️  End-to-end experiment scripts
│   ├── create_index.py            🛠️  Index creation
│   ├── generateworkload.py        🎲  Query generator
│   └── run_workload.py            🚀  Execute evaluation
├── evaluate.py                    🏁 Model evaluation
├── SLM_Like.py                    🏋️ Model training
└── models                         🧠Trained model parameters
```

------

## 📦 Requirements

- 🐍 Python 3.9.13
- 🔥 PyTorch 2.5.1
- 🔢 Numpy 2.0.2
- 📊 Pandas 1.5.1

------

## 📥 Setup & Dataset

For the convenience of demonstration, we have provided a sample of 10,000 records from the TPC-H lineitem table. 😊 
We denote it as `lineitem10000` 📄, which contains  10,000 entries from the `l_comment` column of the TPC-H `lineitem` table. Saved as `lineitem10000.csv` in the `data` directory.More datasets can be referred to through our links. 📚

## 📧 Emails

For the **Emails** dataset, we use `gen_email.py` to generate the data 🛠️. After generation, **deduplication** is required to ensure clean and unique entries 🧹✨.

▶️ Run the following commands to generate and clean the data:

```
python gen_email.py     # 📧 Generate email samples
python duplicate.py     # 🧼 Deduplicate entries
```

✅ Now your email dataset is fresh, clean, and ready to go! 🚀

------

## 📱 Phone Numbers

For the **Phone_numbers** dataset, we use `gen_phone_data.py` to generate the data 📲. Similar to the emails, **deduplication** is necessary after generation to avoid noisy duplicates 🧽.

▶️ Use the following commands:

```
python gen_phone_data.py  # 📱 Generate phone number samples
python duplicate.py       # 🧼 Deduplicate entries
```

📦 Once cleaned, the phone number dataset is good to go for training, testing, or querying! 💪📊

------

## 🧪 Train SMILE

You can use our pre-trained model (W1 setting) ✅ and train your own SMILE with an 8GB budget-friendly GPU to easily verify it. It's super convenient and cost-effective, and you can get started right away. Just plug in the model and see the magic happen! 🌟

```bash
python SLM_Like.py   --lr 0.0003   --batch_size 1024   --inPct 0.1   --pct 0.2   --saveName lineitem10000_lr0.0003_in1Pct2   --data_path ./data/lineitem10000.csv  --GPU 0 
```

💡 *Tip: Adjust `--inPct` and `--pct` to control query inclusion and wildcard percentage.*

------

## 📈 Evaluate SMILE

Evaluate the trained model with:

```bash
python evaluate.py   --PathOfModel ./models/lineitem10000_lr0.0003_in1Pct2/Ep_999_Seq2Seq  --inPct 0.1   --pct 0.2
```

🔍 This runs LIKE queries  for different LIKE Workloads 📊.

------

## 💬 Interactive LIKE Pattern Prediction 🧠✨

We provide an **interactive program** 🕹️ that allows you to input a SQL `LIKE` pattern 🔍 (e.g., `%fox%`, `__quick`, `lazy%`) and get **instant predictions** powered by our SMILE model 😄⚡.

Just type your pattern and hit enter ⌨️—our lightweight neural engine will return the predicted matching results 🎯 in real-time!

You can **exit anytime** by typing `'exit'` or `'q'` ❌👋.

------

### 🧪 Try it Yourself!

Launch the program using this command:

```
python chat_inference.py --data_path "./data/lineitem10000.csv" --PathOfModel "./models/lineitem10000_lr0.0003_in1Pct2/Ep_9999_Seq2Seq"
```

------

### 🧙 What it does:

- 🗣️ **Talk to SMILE**: Enter `LIKE` patterns naturally
- ⚡ **Fast Results**: Get predictions almost instantly
- 🎯 **Real Queries**: Test against actual `lineitem` dataset entries
- 💡 **Smart Matching**: Handles wildcards `%` and `_` with learned intelligence

------

> 🧩 *Behind the scenes*: Your pattern is passed through our small but mighty SMILE model 🤖, which acts as a translator to predict the set of rows that match your pattern—**way faster** than scanning the whole column linearly. 🔥

------

Whether you're debugging, testing queries, or just curious—this mode makes SMILE fun and interactive to explore! 🤓🎉

Ready to chat with your database? 💬📊 Let the LIKE magic begin! ✨🪄

Ready to make your databases **smile**? 😄
 Let neural LIKE acceleration begin! ⚡🧠📚

