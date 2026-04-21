# Slide 13: Geometric Robustness & Angle Degradation

## **The Rotation Challenge**
Standard Vision Transformers (ViT) use absolute positional encodings, which are sensitive to the orientation of the object. When we rotate the target image, the PCK inevitably drops.

---

## **Key Findings from our Analysis:**
*   **Performance Decay:** As the angle increases from 0° to 180°, the matching accuracy decreases. This is expected as the "Semantic Map" rotates while the positional embeddings remain fixed.
*   **The PEFT Advantage:** Even under extreme rotation (e.g., 90° or 180°), our **LoRA and BitFit models consistently outperform the frozen baselines**.
*   **Conclusion:** Fine-tuning the adapter doesn't just improve flat matching scores; it makes the descriptors more resilient to geometric distortions compared to the "out-of-the-box" foundation model.
