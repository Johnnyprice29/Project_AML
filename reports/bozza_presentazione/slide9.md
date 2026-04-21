# Slide 9: The Window Mechanism - Spatial Filtering

## **The Noise Problem**
Global Soft-Argmax often fails because background pixels with accidental high similarity pull the predicted coordinate away from the true object part.

---

## **Static Window vs. Dynamic Adaptive Window**
*   **Static Window:** Crop a fixed $k \times k$ area around the highest similarity point. 
    *   *Limit:* If the object is very small or very large, a fixed window is either too noisy or too restrictive.
*   **Dynamic Adaptive Window (Our Choice):**
    *   **Filtering:** We dynamically select the top-K% of similarity values for each query.
    *   **Isolation:** This creates an "island" of high-confidence scores.
    *   **Why we chose it:** It adapts to the object's scale and prevents background noise from entering the Soft-Argmax calculation, significantly increasing PCK in cluttered environments.
