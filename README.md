# Pedagogy-in-the-Loop: MRBench Taxonomy-based AI Tutor Evaluator (Upload In progress)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

A lightweight, edge-deployable classifier for evaluating AI tutor responses based on the MRBench pedagogical taxonomy. This hybrid TF-IDF + XGBoost + rule-based system achieves real-time inference (<5ms) with competitive accuracy (68.2% OOF accuracy, 0.74 F1-score), making it suitable for smart classroom sensor networks and IoT-enabled educational platforms.

## 📚 Overview

While large language models (LLMs) show promise as AI tutors, their pedagogical quality remains difficult to evaluate at scale. Existing approaches rely on:
- ❌ Binary task completion metrics
- ❌ Ad-hoc human ratings
- ❌ Opaque LLM-as-judge pipelines

**Our solution** operationalizes the MRBench taxonomy by:
1. **Defining a binary Good/Poor label** based on three critical dimensions:
   - Providing Guidance
   - Actionability
   - Coherence
2. **Training a hybrid classifier** combining TF-IDF features with XGBoost
3. **Adding a deterministic rule layer** as a pedagogical safety net for scaffolding phrases
4. **Packaging into an interactive Streamlit web app** for real-time and batch analysis

### Key Results

| Metric | Value |
|--------|-------|
| Out-of-Fold Accuracy | 68.2% |
| F1-Score | 0.740 |
| AUC | 0.712 |
| Inference Latency | **<5ms** (vs. ~7s for local 3B LLM) |
| Cohen's Kappa | 0.312 (Fair agreement with experts) |

## 🚀 Features

- **Real-time Analysis**: Paste a tutor response and get immediate Good/Poor classification
- **Batch Processing**: Analyze up to 100 responses at once for comparative studies
- **Explainable Predictions**: View SHAP-based feature importance, salient n-grams, and rule-based boosts
- **Edge-Ready**: Works offline with <200MB RAM, runs on Raspberry Pi 4/5 and NVIDIA Jetson
- **Privacy-Preserving**: All processing stays local—no API calls required

## 🏗️ Architecture
