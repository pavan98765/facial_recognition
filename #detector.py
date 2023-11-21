import argparse
from collections import Counter
import face_recognition
import pickle
import cv2
from pathlib import Path
from datetime import datetime
import os

# Constants
ENCODINGS_PATH = "output/encodings.pkl"
BOUNDING_BOX_COLOR = (0, 0, 255)  # Blue in BGR format
TEXT_COLOR = (255, 255, 255)  # White in BGR format

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Recognize faces")
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument("--test", action="store_true", help="Test using an image")
parser.add_argument("--live", action="store_true", help="Perform live face recognition")
parser.add_argument("--validate", action="store_true", help="Validate using images")
parser.add_argument(
    "-m",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Which model to use for training: hog (CPU), cnn (GPU)",
)
parser.add_argument("-f", action="store", help="Path to an image with an unknown face")
parser.add_argument(
    "--size_ratio",
    type=float,
    default=1.0,  # You can change the default ratio as needed
    help="Ratio to resize input images for face recognition",
)
parser.add_argument(
    "--tolerance",
    type=float,
    default=0.6,  # Default tolerance value
    help="Face recognition tolerance (default: 0.6)",
)
parser.add_argument(
    "--evaluate", action="store_true", help="Evaluate the model on a test dataset"
)
args = parser.parse_args()


# Function to train and encode faces
def train_encode_faces(
    model: str = "hog", encodings_location: str = ENCODINGS_PATH
) -> None:
    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with open(encodings_location, "wb") as f:
        pickle.dump(name_encodings, f)


# Function to mark the database
def mark_db(name):
    with open("output/database.csv", "r+") as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            time = now.strftime("%H:%M:%S")
            date = now.strftime("%Y-%m-%d")
            f.writelines(f"\n{name},{time},{date}")


# Function to recognize faces
def recognize(image, loaded_encodings, display_image=True):
    image = cv2.resize(
        image,
        None,
        fx=args.size_ratio,
        fy=args.size_ratio,
        interpolation=cv2.INTER_LINEAR,
    )

    input_face_locations = face_recognition.face_locations(image, model=args.m)
    input_face_encodings = face_recognition.face_encodings(image, input_face_locations)

    recognized_names = []

    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = face_match(unknown_encoding, loaded_encodings)
        if name:
            mark_db(name)
        if not name:
            name = "unknown"
        if display_image:
            display(image, bounding_box, name)
        recognized_names.append(name)

    return recognized_names if not display_image else image


# Function to compare faces for a match
def face_match(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding, tolerance=args.tolerance
    )
    votes = Counter(
        name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match
    )
    if votes:
        return votes.most_common(1)[0][0]


# Function to display recognized faces
def display(image, bounding_box, name):
    top, right, bottom, left = bounding_box
    cv2.rectangle(image, (left, top), (right, bottom), BOUNDING_BOX_COLOR, 2)

    # Calculate text position
    text_left = left
    text_bottom = bottom + 20  # Adjust the vertical position of the text

    # Draw a filled rectangle as background for text
    cv2.rectangle(
        image,
        (text_left, bottom),
        (right, text_bottom + 5),
        BOUNDING_BOX_COLOR,
        cv2.FILLED,
    )

    # Put text with improved alignment and readability
    text = name
    font_scale = 0.7  # Adjust font size
    font_thickness = 2  # Adjust font thickness
    font = cv2.FONT_HERSHEY_DUPLEX

    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (right + text_left - text_size[0]) // 2
    text_y = text_bottom

    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        font_scale,
        TEXT_COLOR,
        font_thickness,
        lineType=cv2.LINE_AA,
    )


# Function to validate using images
def validate():
    encodings_location = ENCODINGS_PATH
    with open(encodings_location, "rb") as f:
        loaded_encodings = pickle.load(f)

    # Create the output/validation_processed directory if it doesn't exist
    output_dir = "output/validation_output"

    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            image = cv2.imread(str(filepath))
            image_with_faces = recognize(image, loaded_encodings)

            # Extract the filename without extension
            filename_without_extension = Path(filepath).stem

            # Construct the output file path
            output_file_path = os.path.join(
                output_dir, f"{filename_without_extension}_processed.jpg"
            )

            # Save the processed image
            cv2.imwrite(output_file_path, image_with_faces)
            cv2.namedWindow("Recognized Faces", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Recognized Faces", 800, 600)
            # Display the processed image with recognized faces
            cv2.imshow("Recognized Faces", image_with_faces)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# Function to evaluate the model on a test dataset
def evaluate_model(
    test_dataset_path: str,
    model: str = "hog",
    encodings_location: str = ENCODINGS_PATH,
) -> None:
    with open(encodings_location, "rb") as f:
        loaded_encodings = pickle.load(f)

    correct_predictions = 0
    total_predictions = 0

    for class_folder in Path(test_dataset_path).iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            print(f"Evaluating {class_name}...")
            for image_path in class_folder.rglob("*"):
                if image_path.is_file():
                    total_predictions += 1
                    image = cv2.imread(
                        str(image_path)
                    )  # Read the image using cv2.imread
                    recognized_names = recognize(
                        image=image,
                        loaded_encodings=loaded_encodings,
                        display_image=False,
                    )

                    # If there are recognized names, consider the first one
                    recognized_name = (
                        recognized_names[0] if recognized_names else "unknown"
                    )
                    print(f"Recognized: {recognized_name} , class: {class_name}")

                    # Check if the ground truth class_name is in the recognized names
                    if class_name == recognized_name:
                        correct_predictions += 1

    accuracy = (
        correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
    )
    print(f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")


# Main execution
if __name__ == "__main__":
    if args.train:
        train_encode_faces(model=args.m)
    if args.live:
        encodings_location = ENCODINGS_PATH
        with open(encodings_location, "rb") as f:
            loaded_encodings = pickle.load(f)

        cv2.namedWindow(
            "Live Face Recognition", cv2.WINDOW_NORMAL
        )  # Create a named window
        cv2.resizeWindow(
            "Live Face Recognition", 800, 600
        )  # Set the initial window size

        is_fullscreen = False
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_with_faces = recognize(frame, loaded_encodings)
            cv2.imshow("Live Face Recognition", frame_with_faces)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("f"):
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(
                        "Live Face Recognition",
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN,
                    )
                else:
                    cv2.resizeWindow(
                        "Live Face Recognition", 800, 600
                    )  # Set back to 800x600 window size

        cap.release()
        cv2.destroyAllWindows()

    if args.test:
        image_path = args.f
        image = cv2.imread(image_path)
        encodings_location = ENCODINGS_PATH
        with open(encodings_location, "rb") as f:
            loaded_encodings = pickle.load(f)
        image_with_faces = recognize(image, loaded_encodings)
        cv2.imshow("Recognized Faces", image_with_faces)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if args.validate:
        validate()

    if args.evaluate:
        evaluate_model(
            test_dataset_path="test",
            model=args.m,
            encodings_location=ENCODINGS_PATH,
        )
