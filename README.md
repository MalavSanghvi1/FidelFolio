# FidelFolio Market Cap Growth Forecasting

##  Project Overview
This project tackles the **FidelFolio Deep Learning Challenge**, which involves predicting market capitalization growth across multiple time horizons (1Y, 2Y, 3Y) for Indian companies based on fundamental financial indicators using deep learning models.

---

##  Objective
- Model complex, nonlinear relationships between financial indicators and future market cap.
- Predict growth across **short-term (Target 1)**, **medium-term (Target 2)**, and **long-term (Target 3)**.
- Analyze effectiveness of various model architectures (MLP, LSTM, LSTM + Attention, Transformer).
  

---

##  Dataset
- **Rows**: Company-Year instances  
- **Features**: `Feature1` to `Feature28` (fundamental indicators)  
- **Targets**: `Target 1`, `Target 2`, `Target 3` (market cap growth)

### Preprocessing:
- Missing values imputed using company-wise and global means.
- Winsorization to cap outliers.
- Features standardized using `StandardScaler`.

---

##  Models Implemented

### 1. **Multilayer Perceptron (MLP)**
- Feedforward NN with dropout & ReLU
- Trained for 1000 epochs
- RMSE:
```
Target 1: 107.96
Target 2: 203.92
Target 3: 370.11
```

### 2. **LSTM (Vanilla)**
- Sequence-aware model with final timestep target
- Best at 2000 epochs:
```
Target 1: 22.53
Target 2: 56.57
Target 3: 186.59
```

### 3. **LSTM with Attention**
- Uses soft attention for weighted feature aggregation
- Best at 3000 epochs:
```
Target 1: 11.21
Target 2: 36.21
Target 3: 156.02
```

### 4. **Transformer Encoder**
- Positional encoding + multi-head self-attention
- Best at 1200 epochs:
```
Target 1: 33.35
Target 2: 53.66
Target 3: 111.83
```

---


##  Model Comparison
| Model              | Attention | RMSE T1 | RMSE T2 | RMSE T3 |
|--------------------|-----------|---------|---------|---------|
| MLP                | No        | 107.96  | 203.92  | 370.11  |
| LSTM               | No        | 22.53  | 56.57   | 186.59  |
| LSTM + Attention   | Yes       | 11.21   | 36.21   | 156.02  |
| Transformer        | Yes       | 33.35   | 53.66   | 111.83  |

---

##  Run Instructions
```bash
# Preprocess the data
python dataprocessing.py

Train your preferred model
# Uncomment only the model you want to train and run

```

---




