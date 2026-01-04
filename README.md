# ⚙️ Predictive Maintenance Dashboard

A comprehensive machine learning-based predictive maintenance system built with Streamlit that identifies potential equipment failures before they occur, helping organizations optimize maintenance schedules and reduce costs.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Dashboard Pages](#dashboard-pages)
- [Results](#results)
- [Cost-Benefit Analysis](#cost-benefit-analysis)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

LIVE APP - https://machine-health-ai.streamlit.app/

Traditional maintenance strategies often lead to unnecessary downtime or unexpected failures. This predictive maintenance system enables data-driven decision-making by detecting early warning signs of failure, improving reliability, reducing costs, and optimizing maintenance schedules.

The dashboard provides:
- **Real-time failure predictions** based on operational parameters
- **Interactive visualizations** of model performance
- **Cost-benefit analysis** comparing predictive vs. reactive maintenance
- **Actionable recommendations** for maintenance teams

## ✨ Features

- 🔮 **Interactive Predictions**: Real-time failure prediction with adjustable machine parameters
- 📊 **Model Comparison**: Evaluate three different ML models side-by-side
- 💰 **Cost Analysis**: Calculate potential savings from predictive maintenance
- 📈 **Performance Metrics**: Comprehensive model evaluation with multiple metrics
- 🎨 **Beautiful UI**: Professional dashboard with intuitive navigation
- 🔍 **Risk Assessment**: Color-coded risk levels with detailed recommendations

## 📁 Dataset

**Source**: [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (UCI ML Repository)

- **Size**: 10,000 instances
- **Features**: 6 operational parameters
- **Target**: Binary classification (failure vs. normal operation)
- **Class Distribution**: Highly imbalanced (96.61% normal, 3.39% failure)
- **License**: CC BY 4.0

### Features

| Feature | Description | Range |
|---------|-------------|-------|
| Type | Product quality variant (L/M/H) | Categorical |
| Air Temperature | Operating air temperature | 295.3 - 304.5 K |
| Process Temperature | Process temperature | 305.7 - 313.8 K |
| Rotational Speed | Machine rotational speed | 1168 - 2886 RPM |
| Torque | Applied torque | 3.8 - 76.6 Nm |
| Tool Wear | Accumulated tool wear | 0 - 253 minutes |

## 🤖 Models

### 1. Bagging with Tomek Links (⭐ Recommended)
- **Best overall balance** between precision and recall
- **PR AUC**: 0.82
- **Precision**: 0.85
- **Recall**: 0.71
- **F1-Score**: 0.77
- **Use Case**: Production deployment with limited maintenance resources

### 2. Random Forest with RandomOverSampler
- **Strong performance** with good reliability
- **PR AUC**: 0.78
- **Precision**: 0.85
- **Recall**: 0.65
- **F1-Score**: 0.73
- **Use Case**: Alternative for balanced performance

### 3. Balanced Bagging
- **Highest recall** - catches most failures
- **PR AUC**: 0.61
- **Precision**: 0.28
- **Recall**: 0.88
- **F1-Score**: 0.42
- **Use Case**: When missing failures is extremely costly

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/predictive-maintenance-dashboard.git
cd predictive-maintenance-dashboard
```

2. **Create a virtual environment** (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

4. **Ensure model files are present**
Place these pickle files in the project root:
- `Bagging_binary.pkl`
- `randomforest_model.pkl`
- `BalancedBagging_binary.pkl`

## 💻 Usage

### Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

### Making Predictions

1. Navigate to the **"Make Predictions"** page
2. Adjust sliders to set machine parameters:
   - Product Type (L/M/H)
   - Air Temperature
   - Process Temperature
   - Rotational Speed
   - Torque
   - Tool Wear
3. Select a prediction model
4. Click **"🚀 Predict Failure"**
5. View results with risk assessment and recommendations

## 📊 Dashboard Pages

### 🏠 Home
- Project overview and objectives
- Dataset information
- Model summaries
- Feature descriptions

### 📈 Model Performance
- Performance metrics comparison table
- Interactive visualizations (bar charts, radar charts)
- Confusion matrices for all models
- Key insights and recommendations

### 🔮 Make Predictions
- Interactive input sliders for machine parameters
- Model selection
- Real-time predictions with probability scores
- Risk assessment gauge
- Actionable maintenance recommendations
- Cost impact estimates

### 💰 Cost-Benefit Analysis
- Customizable cost parameters
- Financial comparison of all models
- Savings calculations and ROI metrics
- Interactive cost visualizations
- Scenario-based recommendations

## 📉 Results

### Model Performance Summary

| Model | Precision | Recall | F1-Score | PR AUC | Accuracy |
|-------|-----------|--------|----------|--------|----------|
| Bagging + Tomek Links | 0.85 | 0.71 | 0.77 | **0.82** | 0.99 |
| Random Forest + ROS | 0.85 | 0.65 | 0.73 | 0.78 | 0.98 |
| Balanced Bagging | 0.28 | 0.88 | 0.42 | 0.61 | 0.92 |

### Key Findings

✅ **Bagging + Tomek Links** provides the best balance for production use  
✅ **71% of failures** correctly identified while maintaining high precision  
✅ **99% overall accuracy** with minimal false alarms  
✅ **Significantly outperforms** baseline run-to-failure approach  

## 💰 Cost-Benefit Analysis

### Financial Impact (Sri Lankan Rupees)

| Model | Baseline Cost | Model Cost | Savings | ROI |
|-------|---------------|------------|---------|-----|
| Bagging + Tomek Links | Rs. 1,700,000 | Rs. 535,500 | **Rs. 1,164,500** | 68.5% |
| Random Forest + ROS | Rs. 1,700,000 | Rs. 632,500 | Rs. 1,067,500 | 62.8% |
| Balanced Bagging | Rs. 1,700,000 | Rs. 334,500 | **Rs. 1,365,500** | 80.3% |

### Recommendations

**Scenario A - Maximize Savings:**
- Use **Balanced Bagging** if you have adequate maintenance staff
- Achieves highest ROI (80.3%)
- Handles more inspections but prevents costly breakdowns

**Scenario B - Optimize Efficiency:**
- Use **Bagging + Tomek Links** for limited resources
- Best precision-recall balance
- Fewer false alarms, still significant savings (68.5%)

## 🛠️ Technologies Used

- **Python 3.8+**
- **Streamlit** - Dashboard framework
- **Scikit-learn** - Machine learning models
- **Imbalanced-learn** - Handling class imbalance
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

## 📂 Project Structure

```
predictive-maintenance-dashboard/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── models/                         # Trained model files
│   ├── Bagging_binary.pkl
│   ├── randomforest_model.pkl
│   └── BalancedBagging_binary.pkl
│
├── notebooks/                      # Jupyter notebooks
│   └── model_training.ipynb       # Model training and evaluation
│
├── data/                          # Dataset (not included in repo)
│   └── ai4i2020.csv
│
└── images/                        # Screenshots and images
    └── dashboard_preview.png
```

## 📸 Screenshots

### Home Page
![Home Page](images/home_page.png)

### Prediction Interface
![Prediction](images/prediction_page.png)

### Cost-Benefit Analysis
![Cost Analysis](images/cost_analysis.png)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGithub](https://github.com/yourusername)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the AI4I 2020 dataset
- Streamlit team for the amazing framework
- The open-source community for the ML libraries

## 📧 Contact

For questions or feedback, please reach out:

- Email: your.email@example.com
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- GitHub: [@yourusername](https://github.com/yourusername)

---

⭐ **Star this repo** if you find it helpful!

**Made with ❤️ using Python and Streamlit**
