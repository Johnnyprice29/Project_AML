# Slide 4: Layer-wise Semantic Depth Analysis

## **What do different Layers see?**
In Vision Transformers (ViT), layers capture information at different scales of abstraction:

*   **Layer 4 (Low-Level):** Focuses on "Edge & Texture". It identifies colors and simple shapes. Matching based on L4 fails because two different objects with similar colors would be incorrectly mapped.
*   **Layer 8 (Mid-Level):** Focuses on "Part Anatomy". It begins to recognize structures like legs, wheels, or eyes. PCK improves relative to L4.
*   **Layer 11 (High-Level/Semantic):** Captures the "Identity & Concept". It understands the relationship between parts, allowing for robust matching even under pose variations.

---

## **The Performance Drop**
*   **Why it worsens with fewer layers:** As we move towards the input (lower layers), the receptive field is smaller and the features are less specialized for "Classes". 
*   **Optimal Depth:** Layer 11/12 provides the most stable semantic descriptors, which is why we choose the penultimate layer for the final pipeline.
