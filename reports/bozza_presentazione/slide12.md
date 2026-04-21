# Slide 12: Calibration, Regularization & Temperature

## **The Temperature Scaling ($T$)**
In the Soft-Argmax operation, the temperature $T$ controls the "sharpness" of the coordinate prediction.

---

## **Why it works as Calibration:**
*   **Confidence Reliability:** If $T$ is too high, the model is "unsure" and predicts a central blur. If $T$ is too low, it's overconfident (Hard-Argmax).
*   **Sweet Spot ($T=0.05$):** It calibrates the predicted probability map so that the expected value (the coordinate) truly reflects the highest similarity region.

---

## **Why it works as Regularization:**
*   **Spatial Smoothing:** It forces the model to look at a small *neighborhood* rather than a single pixel.
*   **Noise Suppression:** By weighting pixel neighbors, it effectively regularizes the argmax operator, preventing "jumps" to distant noisy pixels and allowing for sub-pixel precision.
*   **Dual Benefit:** In our case, $T=0.05$ simultaneously stabilizes the training gradients (regularization) and improves the final matching accuracy (calibration).
