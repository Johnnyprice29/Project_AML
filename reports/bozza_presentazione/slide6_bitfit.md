# Slide 6b: BitFit - Bias-only Fine-Tuning

## **What is BitFit?**
BitFit is an extremely sparse fine-tuning strategy that ignores all weight matrices ($W$) and only updates the **bias** terms ($b$) of the network.

---

## **The Efficiency Champion**
Even though it performs slightly worse than LoRA (~77.8% vs ~80.5% PCK), its efficiency is unparalleled:
*   **LoRA (Rank 16):** ~906,000 trainable parameters.
*   **BitFit:** ~82,000 trainable parameters.
*   **Comparison:** BitFit uses roughly **11x fewer** trainable parameters than LoRA, yet achieves competitive semantic results.

---

## **Implementation Logic (The Code)**
One of the main advantages of BitFit is its simplicity. In PyTorch, converting a full model into a BitFit model requires just a few lines of logic:

```python
# Freezing everything except biases
for name, param in model.named_parameters():
    if 'bias' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
```

---

## **Why does it work?**
By only shifting the biases, the model doesn't change the fundamental "features" learned by DINOv2, but it "re-centers" them to be more effective for the specific task of coordinate projection in the spatial grid of SPair-71k.
