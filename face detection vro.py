"""face recognition"""
import cv2
import numpy as np
import os

global_user_id = 1

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)


# Method to generate dataset to recognize a person
def generate_dataset(img, id, img_id):
    # write image in data dir
    cv2.imwrite("data/user." + str(id) + "." + str(img_id) + ".jpg", img)


def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    # Converting image to gray-scale`
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords


# Method to detect the features
def detect(img, faceCascade, img_id):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")
    # If feature is detected, the draw_boundary method will return the x,y coordinates and width and height of rectangle else the length of coords will be 0
    if len(coords) == 4:
        # Updating region of interest by cropping image
        roi_img = img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]
        # img_id to make the name of each image unique
        generate_dataset(roi_img, global_user_id, img_id)

    return img


# Method to train the face recognition model
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Function to get the images and label data
    def get_images_and_labels(path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []

        for image_path in image_paths:
            img = cv2.imread(image_path, 0)
            face_id = int(os.path.split(image_path)[-1].split(".")[1])
            face_samples.append(img)
            ids.append(face_id)

        return face_samples, ids

    # Get training data
    faces, ids = get_images_and_labels("data")

    # Train the model
    recognizer.train(faces, np.array(ids))

    # Save the model to the specified path
    model_path = r"C:\Users\HP\PycharmProjects\pythonProject\trained_model.yml"
    recognizer.save(model_path)
    print("Model saved successfully at:", model_path)


# Method to recognize faces in real-time
def recognize_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    model_file = r"C:\Users\HP\PycharmProjects\pythonProject\trained_model.yml"

    if not os.path.exists(model_file):
        print(f"Error: File '{model_file}' does not exist.")
        return

    recognizer.read(model_file)

    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Initialize and start real-time video capture
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Check if confidence is less than 100 (0 is perfect match)
            if confidence < 100:
                label = "User " + str(id)
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                label = "Unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(frame, str(label), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def collect_user_images():
    global global_user_id
    # Capturing real-time video stream. 0 for built-in web-cams, 0 or -1 for external web-cams
    video_capture = cv2.VideoCapture(0)

    # Initialize img_id with 0
    img_id = 0
    images_collected = 0  # Counter for images collected

    while images_collected < 20:  # Capture only 20 images
        if img_id % 1 == 0:
            print("Collected ", images_collected, " images")
        # Reading image from video stream
        ret, img = video_capture.read()
        if not ret:  # Check if frame was successfully captured
            print("Error: Failed to capture frame from video stream")
            break
        # Call method we defined above
        img = detect(img, faceCascade, img_id)
        # Writing processed image in a new window
        cv2.imshow("face detection", img)
        img_id += 1
        images_collected += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # releasing web-cam
    video_capture.release()
    # Destroying output window
    cv2.destroyAllWindows()


def train_new_user_face():
    # Your existing code for training a new user's face goes here
    global global_user_id
    global_user_id += 1
    collect_user_images()
    train_model()


def call_recognition_model():
    recognize_faces()


def main():
    while True:
        print("Menu:")
        print("1. Train a new user's face")
        print("2. Call the recognition model")
        choice = input("Enter your choice (1 or 2): ")
        if choice == "1":
            train_new_user_face()
        elif choice == "2":
            call_recognition_model()
        else:
            print("Invalid choice. Please enter 1 or 2.")


if _name_ == "_main_":
    main()