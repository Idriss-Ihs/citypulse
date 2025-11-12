# 🌆 CityPulse — Smart City Health Dashboard

CityPulse is a **data-driven digital twin of urban well-being**, built to monitor and visualize the health of cities using **open environmental, meteorological, and air quality data**.  
It fuses multiple public data sources into a unified framework for **real-time analysis, event detection, and trend forecasting**.

---

## Project Overview

This project demonstrates how **Data Science, Machine Learning, and Environmental Intelligence** can converge into a single platform that continuously measures and explains city health.  
Each city’s condition is translated into interpretable indicators and automatically summarized through natural-language event explanations.

CityPulse integrates the following pipelines:

1. **Data Ingestion Layer**
   - Automated collection of meteorological, air quality, and geospatial data from multiple APIs (OpenWeather, OpenAQ, etc.)
   - Scheduled batch ingestion for several major cities
   - Standardized logging and error handling for API consistency

2. **Data Fusion & Processing**
   - Merging heterogeneous datasets (weather, AQI, wind, precipitation) into a unified temporal index
   - Feature scaling and normalization across multiple sources
   - Handling missing data and temporal alignment between feeds

3. **Health Scoring Engine**
   - Computation of daily *sub-scores* (`air_score`, `temp_score`, `precip_score`, `wind_score`)  
   - Aggregation into a 0–100 **City Health Index**
   - Scoring logic interprets each environmental parameter on the same comparable scale

4. **Event Detection & Explanation**
   - Identification of **acute stress periods** (e.g., pollution spikes, abnormal meteorological changes)
   - Automatic generation of natural-language reports explaining *why* city health changed
   - Context-aware text summaries highlight the top drivers behind each anomaly

5. **Predictive Modelling**
   - Prototyped forecasting models to anticipate city health degradation
   - Comparative benchmarking of models for temporal prediction accuracy

---

## Streamlit Dashboard

The interactive Streamlit dashboard provides a **comprehensive real-time interface** for exploring urban health insights:

- **Interactive Map View:**  
  Displays cities as dynamic markers with color-coded health and intensity-based radius.  

- **Daily Metrics Panel:**  
  Color-coded scorecards showing air quality, temperature, precipitation, and wind stress.  
  Intuitive color cues (green → healthy, red → critical).

- **Trend Analytics:**  
  Smooth historical health curve with automatic zoom and rolling average smoothing for trend readability.

- **Event Insight Module:**  
  When anomalies are detected, the system generates human-readable explanations that describe  
  the dominant drivers (e.g., “Air pollution and precipitation stress caused an acute drop in city health”).

---

## Core Competencies Demonstrated

- **Data Engineering:**  
  API integration, batch ingestion, and multi-source synchronization

- **Data Science & Feature Engineering:**  
  Signal normalization, environmental metrics design, composite scoring models

- **Analytical Intelligence:**  
  Event detection through threshold and anomaly tracking, interpretable scoring logic

- **Visualization & Storytelling:**  
  Clean and functional Streamlit architecture with interactive geospatial and temporal layers

- **Machine Learning & Forecasting:**  
  Model comparison for city health prediction and temporal pattern recognition

- **Software Architecture & MLOps:**  
  Modular codebase with versioned configs, structured logging, and reproducible pipelines

---

## Outcome

The final system provides:
- A **functional Smart City digital twin** prototype that transforms raw sensor data into interpretable health analytics  
- A strong proof of concept in **data fusion, environmental intelligence, and explainable analytics**

<p align="center">
  <img src="assets\CityPulse_Dashboard.png" width="800" alt="CityPulse Dashboard Preview">
  <br>
  <em>Interactive Smart City Health Dashboard (CityPulse)</em>
</p>


CityPulse stands as a showcase of **versatility, technical depth, and design sense**, uniting advanced data science with real-world relevance.

---

**CityPulse © 2025 — Environmental Intelligence for Smarter Cities**
