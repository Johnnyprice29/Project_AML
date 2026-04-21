# Slide 15: Future Work - Towards World Models (2025/2026)

## **The Next Frontier: Beyond 2D Correlation**
The field of Semantic Correspondence is shifting from simple feature matching to understanding "World Dynamics".

---

## **Key Research Directions & Thesis Topics:**
1.  **V-JEPA (Joint-Embedding Predictive Architecture):** 
    *   Inspired by **Yann LeCun**, JEPA avoids generative pixels and focuses on Predicting latent representations.
    *   *Thesis idea:* Replacing DINOv2 with V-JEPA backbones to achieve "Causal" semantic consistency that understands object depth and occlusion natively.
2.  **Inference-Time Augmentation (ITA):** Implementing a multi-pass approach where the image is rotated and averaged to eliminate the "Positional Bias" we observed.
3.  **World Models for Correspondence:** 
    *   Learning a internal simulator of the world to "predict" where a part should be even if it's hidden.
    *   Using **Self-Supervised Video Learning** to ensure correspondances are stable across time, not just static pairs.

---

## **Current Trend (2025/26):**
Integration of **Foundation World Models** that learn the physical laws of objects, allowing for correspondence that works even between very different categories (e.g., matching a biological leg to a mechanical robot leg via "Functional" similarity).
