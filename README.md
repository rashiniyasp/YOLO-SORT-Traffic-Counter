# 🚗 YOLO-SORT Traffic Object Counter

This project implements a real-time vehicle counting system using the YOLOv8 object detection model and the SORT (Simple Online and Realtime Tracking) algorithm, integrated with OpenCV for video processing and visualization. The system is designed to count vehicles passing a predefined line within a specified Region of Interest (ROI).

## ✨ Features

*   **Object Detection:** Utilizes Ultralytics YOLOv8l for accurate and efficient detection of various vehicle types (cars, trucks, buses, motorcycles, bicycles).
*   **Object Tracking:** Employs the SORT algorithm to maintain consistent IDs for detected objects across frames, preventing double-counting.
*   **Region of Interest (ROI) Masking:** Processes detections only within a specified area using a custom mask image, focusing on relevant regions.
*   **Counting Line:** Defines a virtual line for counting objects as they cross, ensuring precise vehicle counts.
*   **Real-time Visualization:** Displays bounding boxes, tracking IDs, and the cumulative count directly on the video stream.
*   **Configurable:** Easy to adjust detection confidence, tracking parameters, and ROI/counting line coordinates.

## 🛠️ Technologies Used

*   **Python**
*   **OpenCV:** For video processing, frame manipulation, and visualization.
*   **Ultralytics YOLOv8:** For state-of-the-art object detection.
*   **SORT Algorithm:** For robust multi-object tracking.
*   **`cvzone`:** A helpful library for overlaying text and shapes in OpenCV.

## 📦 Installation & Setup

Follow these steps to get the project up and running on your local machine.

### Prerequisites

*   Python 3.8+
*   Git

### Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/rashiniyasp/YOLO-SORT-Traffic-Counter.git
    cd YOLO-SORT-Traffic-Counter
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    ```
    *   **Activate on Windows:**
        ```bash
        .venv\Scripts\activate
        ```
    *   **Activate on macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download YOLOv8 Model Weights:**
    *   The `yolov8l.pt` model weights are required.
    *   **If Git LFS is set up:**
        The model should be automatically downloaded when you clone the repository. If not, run `git lfs pull`.
    *   **If Git LFS is NOT set up:**
        Download `yolov8l.pt` from the [Ultralytics GitHub Releases page](https://github.com/ultralytics/ultralytics/releases). Place the downloaded file in the root of your `YOLO-SORT-Traffic-Counter` directory.

5.  **Prepare Video and Mask Files:**
    *   **Input Video:** Place your desired input video (e.g., `cars.mp4`) into the `videos/` directory. You can use your own video or a sample traffic video.
    *   **Mask Image:** Ensure `mask.png` is also placed in the `videos/` directory. This mask defines your Region of Interest. If you use a different video, you might need to create a new mask that matches its dimensions and the area you want to monitor.

## 🏃 Usage

Once setup is complete, you can run the main script:

```bash
python traffic_obj_counter.py
