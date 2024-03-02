import cv2
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import StratifiedKFold
import random
import pyttsx3
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os

# Logging configuration
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model():
    logging.info("Start training the model")

    # Getting the current working directory
    current_directory = os.getcwd()

    # Creating the full path to the image file
    image_path = os.path.join(current_directory, 'Pszczolaskrzydelka.jpg')

    # Loading the image
    image = cv2.imread(image_path)

    # Convert image from BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range of yellow color in HSV
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

    # Create mask for yellow color
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Define range of black color in HSV
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([180, 255, 30], dtype=np.uint8)

    # Create mask for black color
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    # Find contours on yellow mask
    contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find contours on black mask
    contours_black, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare training data
    data = []
    labels = []

    # Check if there exists an area that is both yellow and black
    for contour_yellow in contours_yellow:
        for contour_black in contours_black:
            x_yellow, y_yellow, w_yellow, h_yellow = cv2.boundingRect(contour_yellow)
            x_black, y_black, w_black, h_black = cv2.boundingRect(contour_black)

            # Check for overlapping areas
            if x_yellow < x_black + w_black and x_yellow + w_yellow > x_black and y_yellow < y_black + h_black and y_yellow + h_yellow > y_black:
                # Area contains both yellow and black colors

                # Add bee image as training data
                bee_image = image[y_yellow:y_yellow+h_yellow, x_yellow:x_yellow+w_yellow]
                resized_bee_image = cv2.resize(bee_image, (32, 32))
                data.append(resized_bee_image)
                labels.append(1)  # 1 represents a bee

    # Convert data to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Normalize image data
    data = data.astype('float32') / 255.0

    # Convert labels to one-hot encoding
    labels = to_categorical(labels, 2)

    # Define model architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))  # 2 neurons because we have two classes (bees and non-bees)

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Prepare cross-validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Train model using cross-validation
    for train_indices, test_indices in kfold.split(data, labels.argmax(axis=1)):

        random.shuffle(train_indices)
        random.shuffle(test_indices)

        train_data, test_data = data[train_indices], data[test_indices]
        train_labels, test_labels = labels[train_indices], labels[test_indices]

        model.fit(train_data, train_labels, epochs=10, batch_size=64, validation_data=(test_data, test_labels))

    logging.info("Model training finished")
    return model

def track_object(model):
    logging.info("Start object tracking")

    # Initialize camera
    cap = cv2.VideoCapture(0)

    # Initialize GOTURN tracker
    tracker = cv2.TrackerGOTURN_create()

    # Initialize pyttsx3 engine
    engine = pyttsx3.init()

    # Set voice parameters
    engine.setProperty('rate', 150)    # Speaking rate (default 200)
    engine.setProperty('volume', 0.9)  # Volume (from 0 to 1)

    # Initialize list to store bee count in each second
    bee_count = []

    # Initialize time variables
    start_time = None
    current_time = None

    while True:
        # Read frame from camera
        ret, frame = cap.read()

        if not ret:
            break

        # Initialize GOTURN tracker on first frame
        if cv2.waitKey(1) & 0xFF == ord('s'):
            bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
            tracker.init(frame, bbox)
            start_time = cv2.getTickCount()

        # Read current frame time
        current_time = cv2.getTickCount()

        # Calculate duration
        duration = (current_time - start_time) / cv2.getTickFrequency()

        # Check if tracking should be stopped (e.g., after 10 seconds)
        if duration > 10:
            break

        # Update tracker and get new bbox coordinates
        success, bbox = tracker.update(frame)

        # Calculate bee count in current frame and add to list
        bee_count.append(np.random.randint(0, 10))  # Example random value - replace with actual bee count

        # If tracking is successful, draw rectangle around tracked object
        if success:
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            # If tracking fails, send audio message
            text_to_speak = "Tracking error"
            engine.say(text_to_speak)
            engine.runAndWait()

        # Display frame with rectangle
        cv2.imshow("Frame", frame)

        # Break loop if user presses 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()

    # Calculate average bee count per second
    average_bee_count_per_second = sum(bee_count) / len(bee_count)

    # Create DataFrame with bee count in each second
    df = pd.DataFrame({"Time (s)": range(len(bee_count)), "Bee Count": bee_count})

    # Plot the data
    plt.plot(df["Time (s)"], df["Bee Count"])
    plt.xlabel("Time (s)")
    plt.ylabel("Bee Count")
    plt.title(f"Bee Count Over Time (Average: {average_bee_count_per_second:.2f} bees/s)")

    # Save the plot to a file
    plot_filename = "bee_tracking_plot.png"
    plt.savefig(plot_filename)
    logging.info(f"Plot saved to file: {plot_filename}")

    # Generate report
    report_filename = "bee_tracking_report.txt"
    with open(report_filename, "w") as report_file:
        report_file.write(f"Average Bee Count per Second: {average_bee_count_per_second:.2f} bees/s\n\n")
        report_file.write("Bee Count Over Time:\n")
        report_file.write(df.to_string(index=False))
    logging.info(f"Report saved to file: {report_filename}")

    # Provide feedback on the model
    evaluate_model(model, test_data, test_labels)

def evaluate_model(model, test_data, test_labels):
    # Evaluate model on test data
    loss, accuracy = model.evaluate(test_data, test_labels)
    logging.info(f'Loss: {loss}, Accuracy: {accuracy}')

if __name__ == "__main__":
    # Train the model
    trained_model = train_model()

    # Prepare test data (for model evaluation)
    test_data = np.random.rand(100, 32, 32, 3)  # Example test data
    test_labels = to_categorical(np.random.randint(2, size=100), 2)  # Example test labels

    # Track object and evaluate model
    track_object(trained_model)
