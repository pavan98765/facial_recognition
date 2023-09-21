Facial Recognition
This repository contains Python scripts for facial recognition, including training, testing, validation, and real-time recognition with or without object tracking. Here's an overview of each script:

Detector.py:
Detector.py is a versatile script that can perform various facial recognition tasks:

Training and Encoding: Train the model on input data and save encodings for future use.
Testing: Test on a single image to recognize faces.
Validation: Validate on multiple images and save processed images with recognized faces.
Live Recognition: Perform real-time face recognition using a webcam or video feed.

Lite.py:
Lite.py is a simplified version of Detector.py, focused primarily on real-time recognition. It loads pre-trained encodings and attempts to match faces in a live video feed.

Tracking.py:
Tracking.py is a script with two main phases:

Face Recognition Phase: Recognize faces and, when a match is found, initiate tracking.
Object Tracking Phase: Track recognized faces and store tracking data in a database file (CSV).

Lite_tracking.py:
Lite_tracking.py is a more streamlined version of Tracking.py, emphasizing live recognition and tracking.

Video_process.py:
Video_process.py is similar to Detector.py but specifically designed for processing videos. It performs recognition on video frames and saves the processed video output.

Video_process_tracking.py:
Video_process_tracking.py is similar to Tracking.py but tailored for video processing. It combines recognition and tracking on video frames and saves the processed video output.

Dependencies
Ensure you have the following dependencies installed:

Python (>=3.6)
OpenCV (cv2)
face_recognition
Other relevant libraries (specified in individual script requirements)
