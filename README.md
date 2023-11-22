# Facial Recognition Repository

This repository contains Python scripts for facial recognition tasks, offering a versatile set of functionalities for training, testing, validation,image/video processing and real-time recognition, with or without object tracking.

## Repo/Dataset format:

```
face_recognizer/
│
├── output/
│
├── training/
│ └── class1/
│       ├── img_1.jpg
│       └── img_2.png
│ └── class2/
│       ├── img_1.jpg
│       └── img_2.png
|
├── test/
│ └── class_name/
│       ├── img_1.jpg
│       └── img_2.png
│
├── validation/
│       ├── image1.jpg
│       └── image2.jpg
│
├── detector.py
├── image_process.py
└── video_process.py
```

## Scripts Overview:

1. **Detector.py:**
   Detector.py is a versatile script that can perform various facial recognition tasks:

   - **Training and Encoding:** Train the model on input data and save encodings for future use. Place training data (right format) in the "training" folder and run:

   ```
   python Detector.py --train
   ```

   - **Testing:** Test on a single image to recognize faces:

   ```
   python Detector.py --test -f path/to/test_image.jpg
   ```

   - **Validation:** Validate on multiple images and save processed images with recognized faces:

   ```
   python Detector.py --validate
   ```

   - **Live Recognition:** Perform real-time face recognition using a webcam or video feed:

   ```
   python Detector.py --live
   ```

   - **Evaluate:** Evaluate the model on the test dataset:

   ```
   python Detector.py --evaluate
   ```

2. **image_process.py:**
   'image_process.py' processes images from the validation folder, predicts on them, and stores the results in the output folder.

3. **video_process.py:**
   'video_process.py' takes video input, performs facial recognition on frames, and saves the output in the output folder. It only does facial recognition on the frames.

### 4. video_process_tracking.py:

    `video_process_tracking.py` is an enhanced version of `video_process.py` with a two-phase approach: tracking and recognition.

    - **Faster Tracking:**
        This script waits for a match and seamlessly switches to the tracking phase, providing a quicker and more efficient tracking experience.

- **Improved Performance:**
  Designed for tracking a single person, `video_process_tracking.py` optimizes the recognition process, making it faster and more suited to scenarios where tracking is a priority.

- **Two Phases:**
  - **Tracking Phase:** Once a match is found, the script enters the tracking phase, enhancing the accuracy and speed of tracking.
  - **Recognition Phase:** When tracking is lost, the script smoothly transitions back to recognition, ensuring continuous monitoring.

5. **Lite.py:**
   'Lite.py' is a simplified version of 'Detector.py', focusing on real-time recognition. It loads pre-trained encodings and matches faces in a live video feed.

6. **Tracking.py:**
   Tracking.py is a script with two main phases:

   - **Face Recognition Phase:** Recognize faces and, when a match is found, initiate tracking.
   - **Object Tracking Phase:** Track recognized faces and store tracking data in a database file (CSV).

7. **Lite_tracking.py:**
   Lite_tracking.py is a more streamlined version of Tracking.py, emphasizing live recognition and tracking.

## Dependencies:

Ensure you have the following dependencies installed:

- Python (>=3.6)
- OpenCV (cv2) == 4.6.0.66
- face_recognition

Feel free to explore each script for detailed instructions and usage.
