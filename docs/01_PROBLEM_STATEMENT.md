# 01 - Problem Statement

## 1. Executive Summary
Agriculture is the foundation of global food security and economic stability. However, smallholder and medium-scale farmers frequently face devastating crop losses resulting from two compounding challenges:
1. **Undetected or misdiagnosed plant diseases**, leading to delayed or improper chemical intervention.
2. **Inappropriate crop selection or nutrient mismanagement**, where crops are cultivated in soil or microclimates that cannot sustainably support high yields.

Conventional agronomic diagnostic procedures rely either on visual inspection by agricultural extension officers or laboratory soil tests. These procedures are costly, geographically constrained, and often introduce multi-week turnaround delays during critical planting or infection windows.

## 2. Problem Formulation
The goal of this project is to develop an integrated, automated **Farmer Crop Advisory System** that:
- Accurately classifies plant species and leaf pathologies in real-time from mobile or digital camera imagery.
- Analyzes localized environmental readings (Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall).
- Evaluates soil-crop compatibility using a deep neural network conditioned on the detected crop variety.
- Provides actionable, data-driven corrective recommendations to optimize yield and mitigate disease progression.

## 3. Scope and Target Beneficiaries
- **Primary Beneficiaries**: Farmers, agricultural technicians, extension workers, and regional agronomy cooperatives.
- **System Scope**: Real-time image inference covering 29 pathology categories across 9 major agricultural crops, dual-input neural network condition scoring (0–100 scale), and empirical diagnostic deviation breakdown.
