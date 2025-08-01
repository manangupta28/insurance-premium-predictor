# Health Insurance Premium Predictor

A deep learning project that predicts health insurance premiums based on demographic and lifestyle data. Trained with PyTorch and deployed via Streamlit for real-time user interaction.

---

## 🚀 Overview

* **Objective:** Predict annual health insurance charges using a Feedforward Neural Network.
* **Data:** 1,338 records with features: age, sex, BMI, number of children, smoking status, and region.
* **Tech Stack:** Python, Pandas, NumPy, PyTorch, Scikit-learn, Streamlit.

---

## 📊 Data & Preprocessing

1. **Source:** Kaggle Medical Insurance Cost dataset.
2. **Features:**

   * `age` (int)
   * `sex` (binary encoded)
   * `bmi` (float)
   * `children` (int)
   * `smoker` (binary encoded)
   * `region` (one-hot encoded)
3. **Target Transformation:** Log-transform (`np.log`) applied to `charges` to reduce skew.
4. **Scaling:** StandardScaler applied to numerical features for stable training.

---

## 🧠 Model Architecture

* **Framework:** PyTorch
* **Architecture:**

  1. Input layer: 8 features
  2. Hidden layer 1: 64 neurons, ReLU
  3. Hidden layer 2: 32 neurons, ReLU
  4. Output layer: 1 neuron (predicts log(charges))
* **Loss:** Mean Squared Error (MSE)
* **Optimizer:** Adam (lr=0.001)
* **Early Stopping:** Patience = 10 epochs on validation set

---

## 📈 Evaluation

* **Log-space metrics:**

  * MSE: 0.1451
  * RMSE: 0.3810
  * R²: 0.8386
  * MAPE: 2.62%
* **Real-scale metrics (approx):**

  * RMSE: \~₹5,600
  * R²: 0.79

---

## 💻 Deployment

* **Platform:** Streamlit Community Cloud (free)
* **Usage:** Enter your details (age, sex, BMI, children, smoker, region) to get an estimated insurance premium instantly.
* **Link:** [View Live App](https://insurance-premium-predictor-uutxfqqfqw3g3id3sq5knj.streamlit.app/)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to:

* Open an issue to discuss ideas or report bugs.
* Submit a pull request with enhancements.

---



