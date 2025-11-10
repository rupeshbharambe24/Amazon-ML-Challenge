# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** SentinelX
**Team Members:** Rupesh Bharambe, Nilesh Kale, Anjali Maske  
**Submission Date:** 13-10-2025

---

## 1. Executive Summary
We developed a robust product pricing prediction system using advanced feature engineering from catalog content combined with LightGBM regression. Our approach focuses on extracting structured information (Value, Unit, Pack quantities) from unstructured text data and leveraging powerful gradient boosting for accurate price prediction, achieving competitive SMAPE performance without external data sources.

---

## 2. Methodology Overview

### 2.1 Problem Analysis
*The pricing challenge involves predicting product prices based on catalog descriptions and images. Our EDA revealed critical patterns in how product specifications correlate with pricing.*

**Key Observations:**
- Catalog content contains structured patterns (Value, Unit, Pack quantities)
- Product prices follow log-normal distribution, making log transformation beneficial
- Unit conversion is crucial for normalizing product quantities
- Text descriptions contain valuable pricing signals beyond just specifications

### 2.2 Solution Strategy
*Feature engineering-driven approach with gradient boosting*

**Approach Type:** Single Model with Advanced Feature Engineering  
**Core Innovation:** Sophisticated regex-based feature extraction from unstructured text combined with unit normalization and log-space regression

---

## 3. Model Architecture

### 3.1 Architecture Overview
```
Raw Catalog Text → Feature Extraction → Unit Normalization → TF-IDF Vectorization → LightGBM → Price Prediction
                    ↓
               (Value, Unit, Pack)
                    ↓
            Unit Conversion & Total Qty
```

### 3.2 Model Components

**Text Processing Pipeline:**
- **Preprocessing steps:** Regex extraction of Value/Unit/Pack, unit normalization, TF-IDF vectorization
- **Model type:** TF-IDF with 5000 features, n-gram range (1,2)
- **Key parameters:** max_features=5000, ngram_range=(1,2), stop_words='english'

**Feature Engineering Pipeline:**
- **Numeric Features:** Value, Value_base_oz, Total_Qty_oz, Pack, Unit_enc
- **Unit Conversion:** Comprehensive conversion to base ounces for comparability
- **Target Transformation:** Log-price transformation for better regression performance

**Regression Model:**
- **Model type:** LightGBM Regressor
- **Key parameters:** 
  - objective: 'regression'
  - num_leaves: 64
  - learning_rate: 0.05
  - early_stopping_rounds: 50

---

## 4. Feature Engineering

### 4.1 Text Feature Extraction
```python
def extract_value_unit_pack(text):
    # Extracts Value: X, Unit: Y, Pack of Z patterns
    # Handles multiple format variations and edge cases
```

### 4.2 Unit Normalization
- **Weight Units:** grams → ounces, kg → ounces, lb → ounces
- **Volume Units:** ml → fluid ounces, fl oz preserved
- **Count Units:** pieces, tablets, capsules treated as discrete counts
- **Fallback:** Unknown units preserved with original values

### 4.3 Composite Features
- `Total_Qty_oz = Value_base_oz × Pack` - Total normalized quantity
- `price_per_totalqty` - Unit price for training insights
- `log_price` - Transformed target variable

---

## 5. Training Configuration

### 5.1 Data Splitting
- **Training Size:** 80% of 75K samples
- **Validation Size:** 20% of 75K samples
- **Random State:** 42 for reproducibility

### 5.2 Model Training
- **Algorithm:** LightGBM Gradient Boosting
- **Objective:** L2 loss on log-transformed prices
- **Validation:** Early stopping with 50-round patience
- **Final Iterations:** 1019 boosting rounds

---

## 6. Model Performance

### 6.1 Validation Results
- **SMAPE Score:** 51.73%
- **Training L2:** 0.2018
- **Validation L2:** 0.4671

### 6.2 Key Strengths
- **Interpretable:** Feature importance available from LightGBM
- **Efficient:** Fast training and inference
- **Robust:** Handles missing values and unit variations
- **Scalable:** Processes 75K samples efficiently

---

## 7. Conclusion
Our feature engineering approach demonstrates that careful extraction of structured information from unstructured text can achieve competitive pricing prediction performance. The 51.73% SMAPE score validates the effectiveness of unit normalization and log-space regression. Future enhancements would integrate image features for a truly multimodal solution.

---

## Appendix

### A. Code Artefacts
*Complete implementation available in the submitted Colab notebook featuring:*
- Comprehensive data preprocessing pipeline
- Advanced feature engineering functions
- LightGBM model training with validation
- SMAPE evaluation metric
- Production-ready prediction pipeline

### B. Key Implementation Details

**Data Validation:**
- Handles missing values with median imputation
- Robust to unit format variations
- Preserves data integrity through transformations

**Production Readiness:**
- Positive price enforcement via clipping
- Proper CSV output formatting
- Efficient batch processing for 75K test samples

---

**Note:** This solution focuses on text feature extraction and engineering, achieving strong performance through careful data preprocessing and gradient boosting. The approach is fully compliant with competition constraints and demonstrates effective machine learning practices.