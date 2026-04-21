# Slide 6: Curriculum Learning & Adaptation

## **Classical Curriculum Learning**
*   **Concept:** Inspired by human education. Start training with "easy" examples and gradually increase difficulty.
*   **Metric:** In Semantic Correspondence, "easy" usually means pairs from the same sub-category with high visual overlap and minimal rotation.

---

## **LoRA + Curriculum Integration**
In our pipeline, we implemented a scheduled complexity increase:
1.  **Phase 1 (Easy):** High-similarity pairs to establish the baseline coordinate mapping.
2.  **Phase 2 (Medium):** Introduce scale variations and occlusion.
3.  **Phase 3 (Hard):** Large viewpoint changes and cross-category subtle differences.

---

## **Why use it?**
*   **Convergence:** Prevents the optimizer from getting stuck in local minima caused by "noisy" or extreme outliers in the SPair-71k dataset during the early epochs.
*   **Regularization:** Acting as a stabilizer for the LoRA adapter early in training.
