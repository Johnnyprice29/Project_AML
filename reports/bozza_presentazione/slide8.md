# Slide 8: PCK - Percentage of Correct Keypoints

## **Gold Standard of Semantic Correspondence**
PCK is the standard metric used to evaluate how accurately a model identifies corresponding points across different images.

---

## **Mathematical Definition:**
$$PCK@\alpha = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1} \left( \frac{d(\hat{p}_i, p_i)}{\max(H, W)} \leq \alpha \right)$$
*   $p_i$: Ground Truth coordinates.
*   $\hat{p}_i$: Predicted coordinates.
*   $\alpha$: Threshold (Sensitivity).

---

## **Interpreting $\alpha$:**
*   **$\alpha = 0.10$ (Standard):** High-level semantic agreement. If the error is less than 10% of the image size, it is considered correct.
*   **$\alpha = 0.05$ (Strict):** Tests precision and sub-pixel accuracy.
*   **$\alpha = 0.20$ (Loose):** Tests if the model at least identifies the correct general part of the object.
