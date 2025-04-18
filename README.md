# Parkinson’s Voice Screening Dashboard

*A Streamlit web‑app prototype for real‑time detection of Parkinson’s disease from voice‑feature vectors.*

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32-orange)

---

## Overview
This dashboard accompanies the manuscript  
**“Hybrid Sampling and Stacked Machine Learning on Voice Features for Early Parkinson’s Detection.”**

> **Research‑only disclaimer:** The dashboard is provided **solely for research and educational purposes**.  
> It is **not** a certified medical device and must **not** be used to make clinical decisions.

Key capabilities
* **Patient‑level prediction** with calibrated probabilities and confidence labels.  
* **Radar plot** comparing a patient’s top four vocal biomarkers (spread1, spread2, PPE, MDVP:Fo) against cohort means.  
* **Global performance tab** showing confusion matrix and model‑comparison bar chart.  
* **Feature‑importance tab** with ranked biomarker contributions and clinical notes.  
* Automatic **fallback to simulated data** if the trained model or dataset is absent, ensuring reproducibility on any machine.

---

## Demo
![App Demo](demo.gif)

## Quick Start

```bash
git clone https://github.com/<user>/parkinsons-voice-streamlit.git
cd parkinsons-voice-streamlit

python -m venv .venv && source .venv/bin/activate   # Linux/macOS
# .\.venv\Scripts\activate                           # Windows

pip install -r requirements.txt
streamlit run app.py
