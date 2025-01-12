# AlphaPose JavaScript Implementation

## Project Goal
The goal of this project is to implement the AlphaPose pose estimation pipeline entirely in JavaScript.

## Current Implementation

### Video Frame Extraction
- Frames are extracted from input videos using the `ffmpeg` library.
- Extracted frames are saved as PNG files and stored locally.

### Frame Preprocessing
- Extracted frames are loaded as tensors using `TensorFlow.js`.
- Frames are resized and normalized for compatibility with the AlphaPose model.

### Models
- ONNX models for YOLO and AlphaPose are loaded using `onnxruntime-node`.

### JavaScript Implementation
- The pipeline and Python-dependent functions have been rewritten in JavaScript.

## Remaining Tasks
- Rewrite the remaining Python-dependent functions in JavaScript.
