# Slide 3: Backbone Analysis & Baseline Evaluation

## **DINOv2 vs DINOv3: The Semantic Battle**
*   **Performance Similarity:** Both models provide outstanding semantic descriptors. DINOv3 (ViT-g) offers slightly higher resolution and refined attention maps but for the task of *Semantic Correspondence*, the raw feature quality at ViT-B scale is comparable.
*   **Why DINOv2 leads:** Our tests show DINOv2 remains slightly more robust in zero-shot part-level alignment because its training (discriminative SSL) is specifically tuned for local feature consistency.

---

## **The SAM Anomaly: Why the Failure?**
*   **Task Misalignment:** SAM is designed for *Object Segmentation* (defining boundaries). It excels at identifying *where* an object is, but not *what* individual parts represent semantically.
*   **Descriptor Weakness:** SAM’s embeddings are optimized to distinguish pixels within an object from the background, rather than matching a "dog's ear" in Image A to a "dog's ear" in Image B.
*   **Zero-Shot PCK:** Resulting in significantly lower PCK results compared to DINOv2, as SAM lacks the inter-image semantic awareness.
