# Workout Form Analysis Project

This project provides an automated system for analyzing workout form from video input, offering both visual and textual feedback to help users improve their exercise technique. The system leverages state-of-the-art pose estimation and rule-based logic to detect common form deviations for specific exercises.

## Project Overview

### 1. Environment Setup
The initial step involves setting up the development environment by installing essential libraries such as MediaPipe for pose estimation, OpenCV for video processing, NumPy for numerical operations, and Pandas for data manipulation.

### 2. Video Input Handling
Users can upload their 5-10 second workout video clips. The system loads these videos and extracts basic properties like resolution, frame rate, and duration, ensuring the video is ready for further processing.

### 3. Pose Keypoint Detection
MediaPipe Pose, a robust pose estimation model, is utilized to detect and extract 3D keypoints (e.g., shoulders, elbows, wrists, hips) from each frame of the uploaded video. These keypoints provide a detailed representation of the user's body posture and movement.

### 4. Keypoint Data Extraction and Smoothing
The raw 3D keypoint data is then processed. Relevant keypoints are extracted, and smoothing techniques (e.g., rolling mean) are applied to reduce jitter and noise, resulting in a cleaner time-series dataset of body movements.

### 5. Define Exercise-Specific Form Rules
Rule-based logic is developed for specific exercises (e.g., bicep curl). These rules define correct form based on angles (e.g., elbow angle) and alignments (e.g., wrist-shoulder). The rules are configured to identify deviations from ideal form at different stages of the exercise (e.g., start position, peak contraction, full extension).

### 6. Form Evaluation and Feedback Generation
For each frame of the video, the system calculates relevant angles based on the smoothed keypoint data and evaluates them against the predefined exercise-specific rules. Deviations from correct form are identified and recorded frame-by-frame.

### 7. Generate Visual Feedback Video
A new video is created with overlays on the original user's video. This visual feedback includes:
*   Detected keypoints and connections.
*   Lines representing key angles with their real-time values.
*   Textual feedback messages appearing on the screen when form deviations are detected (e.g., 'Elbow too low!', 'Good form!').

### 8. Generate Textual Report
A comprehensive textual report is generated summarizing all detected deviations. This report provides:
*   An overall performance summary.
*   A list of specific issues detected, including the percentage of frames affected by each deviation.
*   Personalized corrective advice for each identified deviation.
*   General recommendations for improvement.
