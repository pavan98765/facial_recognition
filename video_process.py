import pickle
import cv2
import face_recognition
import time

encodings_location = "output/encodings.pkl"
with open(encodings_location, "rb") as f:
    loaded_encodings = pickle.load(f)

# Load known face encodings and names
known_encodings = loaded_encodings["encodings"]  # List of known face encodings
known_names = loaded_encodings["names"]  # List of corresponding names

# Initialize video capture
input_video_path = (
    "C:\\Users\\Jarvis\\Desktop\\BTP_face\\DATA\\#Drone_Data\\videos\\1m.MP4"
)
cap = cv2.VideoCapture(input_video_path)

# Get the frame dimensions and calculate the output video's FPS based on skip_frames
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
skip_frames = 10  # Adjust this value to skip fewer or more frames
size_ratio = 2
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

start_time = time.time()
original_frame_size = (frame_width, frame_height)

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
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1.5  # Adjust the font size as needed
            font_thickness = 3  # Adjust the font thickness as needed
            text_color = (255, 255, 255)  # White color

            cv2.putText(
                frame,
                name,
                (left + 6, top - 10),
                font,
                font_scale,
                text_color,
                font_thickness,
                lineType=cv2.LINE_AA,
            )

    frame = cv2.resize(frame, original_frame_size)
    # Write the modified frame to the output video
    out.write(frame)

    # Verbose message
    print(f"Processed frame {frame_count}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

# Release video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

end_time = time.time()  # Record the end time
processing_time = (end_time - start_time) / 60  # Calculate the processing time
print(f"Processing time: {processing_time:.2f} minutes")
