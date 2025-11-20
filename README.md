# Model Export & Test Tool (pt2onnx)

A Streamlit-based utility for exporting YOLO PyTorch models (`.pt`) to optimized formats (`.onnx`, `.engine`) and testing them on video files. This tool is essential for preparing models for use with **NeedAimBot**.

## Features

*   **Model Export**:
    *   Convert `.pt` to `.onnx` (FP16, Dynamic Batch, Simplified).
    *   Convert `.pt` to `.engine` (TensorRT) - *Requires TensorRT 8.6+*.
    *   Add NMS (Non-Maximum Suppression) module for faster post-processing.
*   **Model Testing**:
    *   Test exported models (`.pt`, `.onnx`, `.engine`) on video files.
    *   Visual feedback with bounding boxes and confidence scores.
    *   Adjustable confidence threshold and input resolution.

## Prerequisites

*   **Python**: 3.10 or higher.
*   **GPU**: NVIDIA GPU with CUDA support (for TensorRT export/inference).
*   **Dependencies**: See `requirements.txt`.

## Installation

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For TensorRT support, ensure you have the correct `tensorrt` python package installed matching your CUDA version.*

2.  **Prepare Models**:
    Place your trained YOLOv8/v11 `.pt` models in the `models/` directory.

## Usage

Run the Streamlit application:

```bash
streamlit run helper.py
```
*Or use the provided batch file:*
```cmd
run_helper.bat
```

The tool will open in your default web browser.

### Export Tab

1.  **Select Model**: Choose a `.pt` file from the list.
2.  **Select Size**: Choose the input resolution (320, 480, or 640).
    *   *Recommendation*: Match this to your `detection_resolution` in NeedAimBot.
3.  **Export Format**:
    *   **ONNX**: Recommended for general compatibility.
    *   **Engine**: For maximum performance on specific hardware.
4.  **Optimization**:
    *   **FP16**: Enable for faster inference (minimal accuracy loss).
    *   **Simplify**: Optimizes the graph structure.
    *   **NMS**: Adds post-processing to the model (recommended for NeedAimBot).
5.  Click **Export model**.

### Tests Tab

1.  **AI Model**: Select an exported model (`.onnx` or `.engine`).
2.  **Video Source**: Use the default test video or upload your own.
3.  **Device**: Select GPU ID (usually `0`).
4.  **Test Detections**: Click to start the visual test.

## Recommended Export Settings for NeedAimBot

*   **Format**: ONNX
*   **Precision**: FP16 (Half)
*   **Simplify**: Yes
*   **NMS**: Yes
*   **Image Size**: 320 or 480 (depending on your GPU power)

## Troubleshooting

*   **TensorRT Errors**: Ensure your Python `tensorrt` version matches your system's TensorRT library version.
*   **Missing Modules**: The script attempts to auto-install missing modules. If it fails, install them manually via pip.
