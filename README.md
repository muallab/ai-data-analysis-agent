# Data Copilot — Ask, Analyze, Visualize

An AI-powered data analysis assistant built with **Streamlit, Pandas, Altair, and OpenAI**.
Upload your dataset, ask questions in plain English, and receive instant **SQL/Pandas plans, execution results, visualizations, and explanations** — all within a single application.

---

## 🚀 Features

* **Upload & Inspect Data**

  * Supports CSV and Excel files (≤200 MB).
  * Automatic schema detection with column types, missing values, and dataset preview.

* **Smart Analysis Suggestions**

  * AI generates relevant analysis questions based on your dataset.
  * Helps guide exploration and insights.

* **Planning Engine**

  * Produces structured JSON plans in `SQL` or `Pandas`.
  * Displays both reasoning and generated code for transparency.

* **Execution Layer**

  * Runs the generated plan against your dataset.
  * Returns accurate tabular results.

* **Interactive Visualizations**

  * Generates bar charts, line charts, and pie charts automatically.
  * Uses Altair for clean and interactive visuals.

* **Plain English Explanations**

  * Summarizes findings in natural language.
  * Provides context, highlights trends, and suggests next steps.

---

## 🛠️ Tech Stack

* [Streamlit](https://streamlit.io/) — Application framework & UI
* [Pandas](https://pandas.pydata.org/) — Data processing
* [Altair](https://altair-viz.github.io/) — Data visualization
* [OpenAI API](https://platform.openai.com/) — AI-powered analysis
* Python 3.9+

---

## ⚡ Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-data-analysis-agent.git
cd ai-data-analysis-agent

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Access the app at [http://localhost:8501](http://localhost:8501).

---

## 📊 Example Workflow

1. Upload `sales.csv`
2. Ask: *“Show total revenue by category as a bar chart”*
3. The app generates a SQL/Pandas plan
4. Results, chart, and explanation are displayed instantly

---

## 🔒 Security & Health

* Runs fully **locally** — no cloud dependencies required.
* Sensitive API keys are stored in `.env` (never hard-coded).
* Minimal external dependencies → easier to maintain.

---

## 🗺️ Roadmap

* ✅ CSV/XLSX uploads, schema summary, analysis suggestions
* ✅ Structured plans with SQL/Pandas
* ✅ Visualizations: bar, line, pie
* ✅ Explanations in plain English
* 🎨 Next: Enhanced UI polish, dark mode, responsive layout
* 📊 Planned: Additional chart types (scatter, histograms)
* 🧑‍💻 Future: Database integration (Postgres, MySQL)

---

## 📜 License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute it.
