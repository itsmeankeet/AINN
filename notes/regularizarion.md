### **Regularization Techniques - Point-wise Summary**

---

#### **1. Overfitting and Underfitting**
- **Overfitting**:
  - The model learns noise and details in the training data.
  - Performs well on the training set but poorly on validation/test sets.
  - Symptoms: High training accuracy but low test accuracy.
  - Cause: Excessive model complexity or too many parameters.
- **Underfitting**:
  - The model is too simple to capture underlying data patterns.
  - Results in low accuracy on both training and test data.
  - Cause: Simplistic models or insufficient training.
- **Goal**: Balance model complexity to generalize well.

---

#### **2. Regularization Techniques**

---

**A. L1 Regularization (Lasso)**  
- **Description**:
  - Adds a penalty equal to the absolute value of coefficients:  
    \( L1\_penalty = \lambda \sum |w| \)
  - \( \lambda \): Regularization parameter controlling penalty strength.
- **Effect**:
  - Encourages sparsity by driving some weights to zero, effectively performing feature selection.
- **Pros**:
  - Useful for feature selection by eliminating irrelevant features.
- **Cons**:
  - Not ideal when all features contribute meaningfully.

---

**B. L2 Regularization (Ridge)**  
- **Description**:
  - Adds a penalty proportional to the square of coefficients:  
    \( L2\_penalty = \lambda \sum w^2 \)
- **Effect**:
  - Reduces the magnitude of weights without setting them to zero.
  - Prevents the model from overly relying on specific features.
- **Pros**:
  - Reduces overfitting; balances the contribution of all features.
- **Cons**:
  - Does not perform feature selection; all weights remain non-zero.

---

**C. Dropout**  
- **Description**:
  - Randomly "drops" (sets to zero) a fraction of neurons during each training pass.
  - Controlled by a probability parameter (e.g., 0.5 for 50% dropout).
- **Effect**:
  - Forces the model to learn redundant representations.
  - Prevents overfitting by introducing randomness and increasing robustness.
- **Pros**:
  - Effective for large networks; improves robustness.
- **Cons**:
  - Adds noise; slows down training; less effective in small networks.

---

**D. Batch Normalization**  
- **Description**:
  - Normalizes layer inputs to stabilize activations during training.
  - Keeps mean and variance consistent by scaling and shifting normalized outputs.
- **Effect**:
  - Reduces internal covariate shift; improves training speed and stability.
  - Acts as a regularizer, often reducing the need for dropout.
- **Pros**:
  - Improves training stability; enables faster learning.
- **Cons**:
  - Adds computational overhead; may not generalize well in all cases.

---

#### **3. Summary of Regularization Techniques**
| **Technique**       | **Effect**                                       | **Pros**                                       | **Cons**                              |
|----------------------|-------------------------------------------------|-----------------------------------------------|---------------------------------------|
| **L1 Regularization**| Drives some weights to zero (sparsity).          | Feature selection.                            | Less effective when all features matter. |
| **L2 Regularization**| Reduces weight magnitudes without zeros.         | Reduces overfitting; balances features.       | Does not perform feature selection.    |
| **Dropout**          | Randomly drops neurons during training.          | Prevents overfitting; increases robustness.   | Adds noise; slows training.            |
| **Batch Normalization**| Normalizes activations for stable learning.   | Improves training stability and speed.        | Computationally expensive.             |

