# Slide 14: Interactive Demonstration with Gradio

## **Moving beyond static Numbers**
Quantitative metrics (PCK) are essential, but visualizing the model's behavior in real-time provides deeper insights into its failure cases and strengths.

---

## **How it works:**
*   **Web-based UI:** Gradio creates a temporary local server (or a public link) to interact with our Python scripts directly from a browser.
*   **Interactive Testing:** 
    *   **Rotation Slider:** Allows users to rotate the target image on-the-fly and observe how keypoint predictions shift.
    *   **Dynamic Pairing:** Users can upload their own images (or select from SPair) to test the model's "World Knowledge".
*   **Real-time Visualization:** The model processes pairs in milliseconds, displaying heatmaps and correspondence arrows instantly.
