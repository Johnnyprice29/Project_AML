# Slide 7: The Evaluation Metrics Module

## **Inside the Metrics Engine**
The matching capability is quantified using a dedicated evaluation module. Its goal is to transform raw similarity maps into interpretable performance scores.

---

## **Key Components:**
1.  **Coordinate Transformation:** Converts high-dimensional grid indices back into image-space (x, y) coordinates.
2.  **Distance Calculation:** Computes the Euclidean distance between the predicted keypoint and the ground truth.
3.  **Normalization:** Since images have different sizes, errors are normalized relative to the object's scale (bounding box or max image dimension).
4.  **Thresholding:** Counts a "hit" only if the error is below a specific percentage ($\alpha$) of the image size.
