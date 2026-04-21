# Slide 11: Fighting Overfitting

## **The Risk of High Capacity**
With 87M parameters, a full fine-tuning would immediately overfit to SPair-71k, learning the specific camera angles or background patterns of that dataset instead of generic matching.

---

## **Our Mitigation Strategies:**
1.  **Parameter Bottleneck (LoRA):** Training only 1% of the model (0.9M parameters) acts as a structural regularizer; it's physically impossible for the model to "memorize" the whole dataset with so few weights.
2.  **Data Augmentation:** Horizontal flipping and color jittering ensure the model focuses on the semantics of parts rather than their exact pixel values or orientation.
3.  **Validation Monitoring:** We save the "Best" model based on validation PCK, not training loss, preventing the model from chasing noise in the final epochs.
