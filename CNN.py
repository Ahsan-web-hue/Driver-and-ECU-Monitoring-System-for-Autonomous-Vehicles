import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = r"D:\FYP\CNN\DATA_SET"
MODEL_PATH = r"D:\FYP\CNN\yawning_model.keras"
RESULT_DIR = r"D:\FYP\CNN\results"

IMG_SIZE = 128
EPOCHS = 20
BATCH_SIZE = 16


# ============================================================
# CREATE RESULT FOLDER
# ============================================================

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

images = []
labels = []

folders = {
    "NotYawning": 0,
    "Yawning": 1
}

print("\n==============================")
print("LOADING DATASET")
print("==============================")

for folder, label in folders.items():

    folder_path = os.path.join(
        DATASET_DIR,
        folder
    )

    if not os.path.exists(folder_path):
        print("Folder not found:", folder_path)
        continue

    for filename in os.listdir(folder_path):

        file_path = os.path.join(
            folder_path,
            filename
        )

        image = cv2.imread(file_path)

        if image is None:
            continue

        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        images.append(image)
        labels.append(label)


# ============================================================
# CONVERT DATA TO NUMPY
# ============================================================

if len(images) == 0:
    print("\nNo images found!")
    exit()

X = np.array(
    images,
    dtype=np.float32
) / 255.0

y = np.array(labels)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n==============================")
print("DATASET INFORMATION")
print("==============================")

print("Total Images :", len(X))
print("Not Yawning  :", np.sum(y == 0))
print("Yawning      :", np.sum(y == 1))


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Images   :", len(X_train))
print("Validation Images :", len(X_val))


# ============================================================
# CNN MODEL
# ============================================================

model = models.Sequential([

    layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3)
    ),

    # Convolution Block 1
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolution Block 2
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolution Block 3
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Classification
    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(
        0.3
    ),

    # Binary output
    layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n==============================")
print("CNN MODEL ARCHITECTURE")
print("==============================")

model.summary()


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n==============================")
print("TRAINING STARTED")
print("==============================")

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    shuffle=True
)


# ============================================================
# VALIDATION
# ============================================================

loss, accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0
)

print("\n==============================")
print("VALIDATION RESULT")
print("==============================")

print(
    "Validation Loss     :",
    round(loss, 4)
)

print(
    "Validation Accuracy :",
    round(accuracy * 100, 2),
    "%"
)


# ============================================================
# SAVE CNN MODEL
# ============================================================

model.save(
    MODEL_PATH
)

print("\n==============================")
print("MODEL SAVED")
print("==============================")

print(MODEL_PATH)


# ============================================================
# FACE DETECTOR
# ============================================================

face_cascade = cv2.CascadeClassifier(

    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


# ============================================================
# IMAGE TESTING
# ============================================================

while True:

    print("\n==============================")

    answer = input(
        "Do you want to examine an image? (yes/no): "
    )

    if answer.lower() != "yes":

        print("\nProgram ended.")
        break


    # ========================================================
    # IMAGE PATH
    # ========================================================

    image_path = input(
        "Enter image path: "
    )

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "Image not found!"
        )

        continue


    # Copy image for drawing results
    display_image = image.copy()


    # ========================================================
    # CNN IMAGE PREPROCESSING
    # ========================================================

    test_image = cv2.resize(

        image,

        (
            IMG_SIZE,
            IMG_SIZE
        )
    )

    test_image = cv2.cvtColor(

        test_image,

        cv2.COLOR_BGR2RGB
    )

    test_image = (
        test_image.astype(
            np.float32
        ) / 255.0
    )

    test_image = np.expand_dims(

        test_image,

        axis=0
    )


    # ========================================================
    # CNN PREDICTION
    # ========================================================

    prediction = model.predict(

        test_image,

        verbose=0

    )[0][0]


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    if prediction >= 0.5:

        status = "YAWNING"

        confidence = (
            prediction * 100
        )

    else:

        status = "NOT YAWNING"

        confidence = (
            (1 - prediction) * 100
        )


    # ========================================================
    # FACE DETECTION
    # ========================================================

    gray = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(

        gray,

        scaleFactor=1.1,

        minNeighbors=5
    )


    # ========================================================
    # DEFAULT VALUES
    # ========================================================

    mouth_width = 0
    mouth_height = 0
    MAR = 0
    level = "UNKNOWN"


    # ========================================================
    # FACE + MOUTH DETECTION
    # ========================================================

    if len(faces) > 0:

        # Select largest face
        x, y, w, h = max(

            faces,

            key=lambda f:
            f[2] * f[3]
        )


        # ----------------------------------------------------
        # FACE RECTANGLE
        # ----------------------------------------------------

        cv2.rectangle(

            display_image,

            (x, y),

            (x + w, y + h),

            (255, 0, 0),

            2
        )


        # ----------------------------------------------------
        # MOUTH REGION
        # ----------------------------------------------------

        mouth_x1 = int(
            x + 0.20 * w
        )

        mouth_x2 = int(
            x + 0.80 * w
        )

        mouth_y1 = int(
            y + 0.55 * h
        )

        mouth_y2 = int(
            y + 0.90 * h
        )


        # Make sure coordinates are valid

        mouth_x1 = max(
            0,
            mouth_x1
        )

        mouth_y1 = max(
            0,
            mouth_y1
        )

        mouth_x2 = min(
            image.shape[1],
            mouth_x2
        )

        mouth_y2 = min(
            image.shape[0],
            mouth_y2
        )


        mouth = gray[
            mouth_y1:mouth_y2,
            mouth_x1:mouth_x2
        ]


        # ----------------------------------------------------
        # MOUTH MEASUREMENT
        # ----------------------------------------------------

        if mouth.size > 0:

            blur = cv2.GaussianBlur(

                mouth,

                (5, 5),

                0
            )


            _, threshold = cv2.threshold(

                blur,

                70,

                255,

                cv2.THRESH_BINARY_INV
            )


            contours, _ = cv2.findContours(

                threshold,

                cv2.RETR_EXTERNAL,

                cv2.CHAIN_APPROX_SIMPLE
            )


            if contours:

                largest = max(

                    contours,

                    key=cv2.contourArea
                )


                if cv2.contourArea(
                    largest
                ) > 20:

                    mx, my, mw, mh = (
                        cv2.boundingRect(
                            largest
                        )
                    )

                    mouth_width = mw

                    mouth_height = mh


                    if mouth_width > 0:

                        MAR = (
                            mouth_height /
                            mouth_width
                        )


        # ----------------------------------------------------
        # YAWNING LEVEL
        # ----------------------------------------------------

        if MAR >= 0.45:

            level = "HIGH"

        elif MAR >= 0.25:

            level = "MEDIUM"

        else:

            level = "LOW"


        # ----------------------------------------------------
        # MOUTH REGION RECTANGLE
        # ----------------------------------------------------

        cv2.rectangle(

            display_image,

            (
                mouth_x1,
                mouth_y1
            ),

            (
                mouth_x2,
                mouth_y2
            ),

            (0, 255, 0),

            2
        )


    # ========================================================
    # TERMINAL RESULT
    # ========================================================

    print("\n==============================")
    print("YAWNING DETECTION RESULT")
    print("==============================")

    print(
        "Image        :",
        image_path
    )

    print(
        "Prediction   :",
        status
    )

    print(
        "Confidence   :",
        f"{confidence:.2f}%"
    )

    print(
        "Mouth Width  :",
        f"{mouth_width:.1f}px"
    )

    print(
        "Mouth Height :",
        f"{mouth_height:.1f}px"
    )

    print(
        "MAR          :",
        f"{MAR:.3f}"
    )

    print(
        "Yawning Level:",
        level
    )

    print("==============================")


    # ========================================================
    # WRITE RESULTS ON IMAGE
    # ========================================================

    cv2.putText(

        display_image,

        f"Status: {status}",

        (20, 35),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (0, 0, 255),

        2
    )


    cv2.putText(

        display_image,

        f"Confidence: {confidence:.1f}%",

        (20, 70),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 0),

        2
    )


    cv2.putText(

        display_image,

        f"Mouth W: {mouth_width:.1f}px",

        (20, 105),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (0, 255, 255),

        2
    )


    cv2.putText(

        display_image,

        f"Mouth H: {mouth_height:.1f}px",

        (20, 140),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (0, 255, 255),

        2
    )


    cv2.putText(

        display_image,

        f"MAR: {MAR:.3f}",

        (20, 175),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2
    )


    cv2.putText(

        display_image,

        f"Yawning Level: {level}",

        (20, 210),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 0),

        2
    )


    # ========================================================
    # SAVE TEST RESULT
    # ========================================================

    original_name = os.path.basename(
        image_path
    )

    name, extension = os.path.splitext(
        original_name
    )

    output_path = os.path.join(

        RESULT_DIR,

        name + "_result" + extension
    )


    cv2.imwrite(

        output_path,

        display_image
    )


    print("\nResult image saved at:")

    print(output_path)


    # ========================================================
    # SHOW RESULT
    # ========================================================

    cv2.imshow(

        "Yawning Detection",

        display_image
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()