# 🎵 AI Influencers Multimodal Analysis

<div align="center">

**Comprehensive Multimodal Analysis of Virtual Influencers' Engagement Dynamics**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)
[![Research](https://img.shields.io/badge/Status-Research%20Project-green.svg)](.)
[![Data](https://img.shields.io/badge/Dataset-Custom%20Annotated-orange.svg)](.)

</div>

---

## 🎯 Overview

This cutting-edge research project investigates engagement patterns in virtual influencer content through **multimodal analysis** of social media data. Utilizing a **custom-scraped and meticulously annotated dataset**, we decode the relationship between audio features and audience engagement across digital platforms.

Our sophisticated analytical framework combines machine learning insights with interactive visualizations to reveal hidden patterns in virtual influencer performance, serving as a foundation for predictive engagement modeling.

> **Current Focus**: Audio label performance analysis  
> **In Development**: Video feature extraction  
> **Focus**: Multi-modal engagement prediction models

## ✨ Key Features

### 📊 **Advanced Analytics Dashboard**
- **Creator-Centric Analysis**: Deep-dive into individual influencer performance metrics with comprehensive label breakdowns
- **Performance Benchmarking**: Identify top performers across multiple engagement dimensions (reach, traction, engagement rate, addictive factor)
- **Cross-Metric Intelligence**: Sophisticated overlap analysis revealing multi-dimensional performance patterns
- **Label Performance Insights**: Win-rate analysis and comparative performance across audio categories
- **Interactive Visualizations**: Dynamic charts, heatmaps, and statistical breakdowns powered by Plotly

### 🔬 **Methodology**
- Statistical significance testing for performance comparisons
- Outlier detection and data quality controls
- Engagement rate normalization using creator-specific baselines
- Follower-normalized metrics for fair cross-creator comparisons

## 📁 Project Structure

```
AI-Influencers-Multimodal-Analysis/
│
├── 🎛️ streamlit_analysis.py          # Interactive dashboard application
├── 📊 final_metadata.csv             # Primary dataset (custom scraped & annotated)
├── 👥 followers-count.csv            # Supplementary follower metrics
├── 📋 requirements.txt               # Python dependencies
└── 📖 README.md                      # Project documentation
```

### 📚 **Dataset Description**
| Component | Description |
|-----------|-------------|
| **Primary Dataset** | Custom-scraped multimodal content with manual annotation of audio features |
| **Engagement Metrics** | Likes, comments, views, replay ratios, and derived engagement indicators |
| **Creator Profiles** | Follower counts, content frequency, and performance baselines |
| **Audio Labels** | Professionally annotated audio categories for content classification |

## 🔬 Research Context

This project represents a comprehensive **academic research initiative** focused on understanding the dynamics of virtual influencer engagement through computational social science methods.

### 🎯 **Research Objectives**
1. **Multimodal Feature Analysis**: Quantifying the impact of audio and visual elements on engagement
2. **Engagement Prediction**: Developing ML models to forecast content performance
3. **Creator Benchmarking**: Establishing performance metrics for virtual influencer assessment
4. **Platform Dynamics**: Understanding algorithmic and audience behavior patterns

## 🚀 Quick Start

### 📦 **Installation**
```bash
# Clone the repository
git clone https://github.com/Xhades14/AI-Influencers-Multimodal-Analysis.git
cd AI-Influencers-Multimodal-Analysis

# Install dependencies
pip install -r requirements.txt
```

### 🎛️ **Launch Dashboard**
```bash
# Start the interactive analytics dashboard
streamlit run streamlit_analysis.py

# Access at: http://localhost:8501
```

## 📈 Usage Guide

### 🎯 **Navigation Overview**
1. **Creator Analysis**: Select from 20+ virtual influencers for detailed performance insights
2. **Top Performers**: Analyze highest-performing content across multiple metrics
3. **Cross-Metric Intelligence**: Discover overlap patterns between different success metrics
4. **Label Comparison**: Compare audio label effectiveness and win rates

### 🛠️ **Advanced Controls**
- **Minimum Reel Threshold**: Filter creators by content volume for statistical reliability
- **Exclude Outliers**: Remove statistical anomalies for cleaner analysis
- **Dynamic Filtering**: Real-time data exploration with interactive controls

> **💡 Pro Tip**: Use the sidebar controls to customize your analysis parameters and exclude creators with skewed data for more representative insights.