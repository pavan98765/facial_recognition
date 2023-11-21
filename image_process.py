import face_recognition
import cv2
import pickle
from pathlib import Path

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")


def recognize_faces_cv2(
    image_location,
    model="hog",
    encodings_location=DEFAULT_ENCODINGS_PATH,
    output_path=None,
    tolerance=0.5,
    size_ratio=1.0,
):
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = cv2.imread(image_location)
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Resize input image
    input_image_resized = cv2.resize(
        input_image_rgb,
        None,
        fx=size_ratio,
        fy=size_ratio,
        interpolation=cv2.INTER_LINEAR,
    )

    input_face_locations = face_recognition.face_locations(
        input_image_resized, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image_resized, input_face_locations
    )

    for bounding_box_resized, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        # Scale back the bounding box to the original size
        bounding_box = [int(coord / size_ratio) for coord in bounding_box_resized]

        boolean_matches = face_recognition.compare_faces(
            loaded_encodings["encodings"], unknown_encoding, tolerance=tolerance
        )
        if any(boolean_matches):  # Check if there is at least one match
            name = loaded_encodings["names"][boolean_matches.index(True)]
            face_distance = face_recognition.face_distance(
                [loaded_encodings["encodings"][boolean_matches.index(True)]],
                unknown_encoding,
            )[0]

            top, right, bottom, left = bounding_box
            color = (0, 255, 0)  # BGR color format

            # Draw the bounding box
            cv2.rectangle(input_image, (left, top), (right, bottom), color, 2)

            # Draw the name label
            label_background = (0, 0, 255)  # Red background
            label_text_color = (255, 255, 255)  # White text color

            # Calculate the position of the label
            label_y = top - 15 if top - 15 > 15 else top + 15

            # Put the name on the image
            cv2.putText(
                input_image,
                f"{name} {face_distance:.2f}",
                (left, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                label_text_color,
                3,
            )

    if output_path:
        cv2.imwrite(
            str(output_path.absolute()), input_image
        )  # Save the processed image


def validate(model="hog", tolerance=0.5, size_ratio=1.0):
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            output_path = (
                Path("output/validation_output") / filepath.name
            )  # Define the output path
            recognize_faces_cv2(
                image_location=str(filepath.absolute()),
                model=model,
                output_path=output_path,
                tolerance=tolerance,
                size_ratio=size_ratio,
            )


if __name__ == "__main__":
    validate(model="hog", tolerance=0.5, size_ratio=1.0)
