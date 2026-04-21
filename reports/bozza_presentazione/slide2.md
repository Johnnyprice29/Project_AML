# Slide 2: Tech Stack & Development Environment

## **Development Tools**
*   **VS Code:** Primary IDE for local development, script refactoring, and code management.
*   **Google Colab:** Cloud GPU environment utilized for heavy model training (LoRA/BitFit) and large-scale evaluation on SPair-71k.

---

## **Deep Learning Libraries**
*   **PyTorch:** Core framework for tensor operations and gradient-based optimization.
*   **Torchvision:** Used for image transformations, augmentation, and data handling.
*   **Torch Hub:** Essential for loading pre-trained foundation models dynamically.
*   **Gradio:** Framework used to build the interactive UI for real-time semantic matching demonstrations.

---

## **Data & Backbone Sources**
### **Datasets:**
*   **SPair-71k:** Main benchmark dataset (71,000 image pairs). Downloaded from [official site/GitHub]. Selected for its focus on semantic rather than just geometric matching.
*   **PF-Pascal:** Used for Out-of-Distribution (OOD) generalization tests.

### **Backbones:**
*   **DINOv2 & DINOv3 (Facebook/Meta Research):** Models loaded via `facebookresearch/dinov2` on Torch Hub.
*   **SAM (Segment Anything Model):** Used as a zero-shot baseline to compare segmentation masks vs. feature extraction.
