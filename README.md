# DataPrep AI 🤖

**Autonomous Data Science Assistant for Instant Data Preparation.**

DataPrep AI is a CLI tool designed to transform raw, messy datasets into clean, ML-ready data with a single command. It automatically handles missing values, encodes categories, normalizes features, and detects anomalies.

## 🚀 Key Features

- **Autonomous Cleaning**: Smart handling of missing values (mean/median/mode) and duplicate removal.
- **Intelligent Encoding**: Automated One-Hot and Label encoding based on feature cardinality.
- **Goal-Oriented Prep**: Optimization for `prediction`, `classification`, or `analysis`.
- **Date Parsing**: Automatically detects and transforms date-like strings to datetime objects.
- **Modern CLI**: Beautiful, colorized logs and progress indicators.

## 📦 Installation

```bash
pip install ibrahim-dataprep
```

## 🛠 Usage

### Auto Clean
Analyze and clean a dataset with one command:
```bash
dataprep auto your_data.csv
```

### Goal-Specific Preparation
Prepare data specifically for machine learning:
```bash
dataprep auto your_data.csv --goal prediction
```

### Custom Fixes (Experimental)
Interact with your data using natural language:
```bash
dataprep fix "fix the missing values in this file" your_data.csv
```

## 📄 License
MIT License. See [LICENSE](LICENSE) for details.
