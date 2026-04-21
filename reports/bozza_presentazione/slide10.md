# Slide 10: The Generalization Gap

## **Training vs. Real World**
A model that performs perfectly on its training set but fails on new data is not useful. The "Generalization Gap" is the performance difference between the Training domain (SPair-71k) and the Test domain (PF-Pascal).

---

## **Key Insights from our Project:**
*   **Zero-Shot Baseline:** Has a huge gap because it hasn't seen the specific categories of SPair.
*   **Finetuned LoRA:** While it specializes on SPair, our evaluation on **PF-Pascal** shows it *still* outperforms the baseline.
*   **Universal Semantic:** This proves the model hasn't just "memorized" SPair images, but has actually learned a more general concept of "Semantic Parts" that transfers across different datasets.
