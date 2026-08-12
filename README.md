
# YOLOv5 Traffic Object Detection Streamlit App

This repository contains a Streamlit application for detecting traffic objects using a pre-trained YOLOv5 nano model.

## Project Structure

```
Deployment/
├── app.py
├── yolov5_traffic_detection_best.pt
├── requirements.txt
├── classes.txt
└── README.md
```

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd Deployment
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .env\Scriptsctivate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

    This will open the application in your web browser.

## Deployment on Streamlit Cloud

To deploy this application on Streamlit Cloud:

1.  Push this `Deployment` folder (and its contents) to a GitHub repository.
2.  Go to [Streamlit Cloud](https://share.streamlit.io/).
3.  Click on "New app" and connect it to your GitHub repository.
4.  Specify the main file path as `app.py` and the Python version.
5.  Deploy the app!

## Model Details

The model used is a YOLOv5 nano (`yolov5n.pt`) trained on the "Traffic Road Object Detection" dataset. The weights `yolov5_traffic_detection_best.pt` are included.

## Class Names

The `classes.txt` file contains the names of the objects the model is trained to detect, which in this case is:
- car

Feel free to contribute or suggest improvements!
