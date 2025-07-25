# Optimal Quantum Reservoir Computing for Market Forecasting: an Application to Fight Food Price Crises
This repository accompanies the paper:  *Optimal Quantum Reservoir Computing for Market Forecasting: An Application to Fight Food Price Crises*

Quantum Reservoir Computing (QRC) is emerging as a powerful paradigm for time series forecasting in the noisy-intermediate scale quantum era. This work explores how carefully designed quantum reservoirs can offer exponential resource efficiency and superior performance for market forecasting tasks, with a focus on agricultural commodity prices like zucchini.

---
## Repository Contents

- **`QRC.py`**  
  Implements the `QuantumRC` class, which includes:
  - Circuit construction using different gate families (e.g., G3, Ising)
  - State evolution and measurement
  - Readout and forecast with and without regressor variables

- **`zucchini_QRC.py`**  
  Runs forecasting experiments on real-world **zucchini price data**. The pipeline includes:
  - Data loading and processing
  - QRC training and predictions
  - Evaluation of the results with mean absolute error and market direction accuracy