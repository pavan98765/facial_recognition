import pickle
import cv2
import face_recognition

encodings_location = "output/encodings.pkl"
with open(encodings_location, "rb") as f:
    loaded_encodings = pickle.load(f)

# Load known face encodings and names
known_encodings = loaded_encodings["encodings"]  # List of known face encodings
known_names = loaded_encodings["names"]  # List of corresponding names

# Initialize video capture
input_video_path = "data/will_chad.mp4"
cap = cv2.VideoCapture(input_video_path)

# Get the frame dimensions and calculate the output video's FPS based on skip_frames
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
skip_frames = 5  # Adjust this value to skip fewer or more frames
size_ratio = 1.0
tolerance = 0.5

original_fps = cap.get(cv2.CAP_PROP_FPS)  # Get the original video's FPS
if skip_frames == 0:
    output_fps = original_fps
else:
    output_fps = max(round(original_fps / skip_frames), 1)

output_video_path = "output/output_video.mp4"
out = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*"XVID"),
    output_fps,
    (frame_width, frame_height),
)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % skip_frames != 0:
        continue  # Skip this frame

    if size_ratio != 1.0:
        frame = cv2.resize(frame, None, fx=size_ratio, fy=size_ratio)

    # Find face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare face encoding with known encodings
        matches = face_recognition.compare_faces(
            known_encodings, face_encoding, tolerance=tolerance
        )

        # If a match is found, choose the name of the known face
        if True in matches:
            matched_index = matches.index(True)  # Get the index of the first match
            name = known_names[matched_index]

            # Draw bounding box and name on the frame for known faces
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1
            )

    # Write the modified frame to the output video
    out.write(frame)

# Release video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
