import face_recognition
import cv2
import pickle
import time

video_input = "C:\\Users\\Jarvis\\Desktop\\BTP_face\\DATA\\#Drone_Data\\videos\\2m.MP4"
output_file = "output/output_tracking.mp4"
encodings_location = "output/encodings_Harsh.pkl"

# Define parameters
skip_frames = 10  # Number of frames to skip before processing
size_ratio = 2.0  # Resize ratio for input frames
tolerance = 0.45  # Tolerance for face recognition


def main():
    # Load known face encodings and names
    with open(encodings_location, "rb") as f:
        loaded_encodings = pickle.load(f)

    known_encodings = loaded_encodings["encodings"]  # List of known face encodings
    known_names = loaded_encodings["names"]  # List of corresponding names

    # Initialize video capture and output writer
    cap = cv2.VideoCapture(video_input)
    # matching the pace of the video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    if skip_frames == 0:
        output_fps = original_fps
    else:
        output_fps = max(round(original_fps / skip_frames), 1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, output_fps, (frame_width, frame_height))

    # Initialize object tracker
    tracker = cv2.TrackerKCF_create()

    tracking = False
    frame_count = 0

    start_time = time.time()
    original_frame_size = (frame_width, frame_height)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % skip_frames != 0:
            continue  # Skip processing this frame

        if size_ratio != 1.0:
            frame = cv2.resize(frame, None, fx=size_ratio, fy=size_ratio)

        if not tracking:
            # Find face locations and encodings in the current frame
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                # Compare face encoding with known encodings
                matches = face_recognition.compare_faces(
                    known_encodings, face_encoding, tolerance=tolerance
                )

                # Check if any match is found
                if any(matches):
                    matched_index = matches.index(True)
                    name = known_names[matched_index]
                    top, right, bottom, left = face_location
                    tracking_bbox = (left, top, right - left, bottom - top)
                    tracking = True
                    tracker.init(frame, tracking_bbox)
                    break  # Break out of the loop once a face is recognized

        else:
            success, tracking_bbox = tracker.update(frame)
            if success:
                # Tracking successful, draw the tracking box
                x, y, w, h = map(int, tracking_bbox)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Draw the name at the bottom of the tracking box
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    frame, name, (x, y - 20), font, 2, (0, 255, 0), 4, cv2.LINE_AA
                )
            else:
                # Tracking lost, switch back to face recognition
                tracking = False
                tracker = cv2.TrackerKCF_create()

        frame = cv2.resize(frame, original_frame_size)
        # Write the frame to the output video
        out.write(frame)

        print(f"Processed frame {frame_count}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    # Release video capture and writer
    cap.release()
    out.release()
    end_time = time.time()  # Record the end time
    processing_time = (end_time - start_time) / 60  # Calculate the processing time
    print(f"Processing time: {processing_time:.2f} minutes")


if __name__ == "__main__":
    main()
