# Slide 5: LoRA - Low-Rank Adaptation Theory

## **The Efficiency Problem**
Fine-tuning a Foundation Model like DINOv2 (87M+ parameters) is computationally expensive and risks destroying the pre-trained weights (*Catastrophic Forgetting*).

---

## **The LoRA Mechanism**
Instead of updating the full weight matrix $W$, we decompose the update into two smaller matrices:
$$W_{updated} = W_{frozen} + (A \times B)$$
*   **A and B** are low-rank matrices.
*   **Rank (r=16):** We found that a rank of 16 is sufficient to capture relevant semantic shifts without over-parameterization.

---

## **Key Benefits:**
1.  **Storage:** We only save the small LoRA matrices (0.9M params vs 87M).
2.  **Stability:** The original DINO knowledge remains frozen, providing a solid foundation for the adapter.
3.  **Speed:** Much faster convergence on Colab GPUs.
