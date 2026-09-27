import os
import cv2
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = r"D:\FYP\CNN\Main_Data\train"
MODEL_PATH = r"D:\FYP\CNN\yawning_model.keras"
RESULT_DIR = r"D:\FYP\CNN\results"

IMG_SIZE = 128
EPOCHS = 20
BATCH_SIZE = 16
RANDOM_STATE = 42

GRAPH_DIR = os.path.join(RESULT_DIR, "graphs")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# FACE DETECTOR
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise SystemExit("ERROR: Haar Cascade could not be loaded.")


# ============================================================
# FUNCTION: DETECT FACE AND CROP
# ============================================================

def detect_and_crop_face(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:

        # Try histogram equalization
        gray_equalized = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray_equalized,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(40, 40)
        )

    if len(faces) == 0:
        return None, None

    # Select largest face
    x, y, w, h = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    image_h, image_w = image.shape[:2]

    # --------------------------------------------------------
    # ADD PADDING AROUND FACE
    # --------------------------------------------------------

    pad_x = int(0.15 * w)
    pad_top = int(0.20 * h)
    pad_bottom = int(0.20 * h)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_top)

    x2 = min(image_w, x + w + pad_x)
    y2 = min(image_h, y + h + pad_bottom)

    face_crop = image[y1:y2, x1:x2]

    return face_crop, (x, y, w, h)


# ============================================================
# LOAD DATASET
# ============================================================

X = []
y = []

classes = {
    "no_yawn": 0,
    "yawn": 1
}

print("\n========================================")
print("LOADING DATASET")
print("========================================")

for folder, label in classes.items():

    path = os.path.join(DATASET_DIR, folder)

    if not os.path.exists(path):

        print("Folder not found:", path)
        continue

    print("\nLoading:", folder)

    image_count = 0
    face_count = 0

    for file in os.listdir(path):

        file_path = os.path.join(path, file)

        img = cv2.imread(file_path)

        if img is None:
            continue

        image_count += 1

        # ----------------------------------------------------
        # DETECT FACE
        # ----------------------------------------------------

        face, box = detect_and_crop_face(img)

        if face is None:

            # If no face detected, skip image
            continue

        face_count += 1

        # ----------------------------------------------------
        # RESIZE FACE
        # ----------------------------------------------------

        face = cv2.resize(
            face,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA
        )

        # BGR -> RGB
        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        X.append(face)
        y.append(label)

    print("Images found :", image_count)
    print("Faces used   :", face_count)


# ============================================================
# CHECK DATASET
# ============================================================

if len(X) == 0:
yes
    raise SystemExit(
        "\nERROR: No valid face images found.\n"
        "Check your dataset and Haar Cascade."
    )


# Convert to NumPy
X = np.array(X, dtype=np.float32) / 255.0
y = np.array(y, dtype=np.int32)

print("\n========================================")
print("DATASET INFORMATION")
print("========================================")

print("Total images :", len(X))
print("Not Yawning  :", np.sum(y == 0))
print("Yawning      :", np.sum(y == 1))

if len(np.unique(y)) < 2:

    raise SystemExit(
        "ERROR: Both classes (no_yawn and yawn) are required."
    )


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n========================================")
print("DATA SPLIT")
print("========================================")

print("Training   :", len(X_train))
print("Validation :", len(X_val))


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([

    # Handles left/right position/orientation
    layers.RandomFlip(
        mode="horizontal"
    ),

    # Small head tilt
    layers.RandomRotation(
        0.08
    ),

    # Small zoom
    layers.RandomZoom(
        height_factor=0.15,
        width_factor=0.15
    ),

    # Face can appear slightly left/right/up/down
    layers.RandomTranslation(
        height_factor=0.10,
        width_factor=0.10
    ),

    # Different lighting conditions
    layers.RandomContrast(
        0.15
    )

], name="data_augmentation")


# ============================================================
# CNN MODEL
# ============================================================

model = models.Sequential([

    layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3)
    ),

    # Data augmentation
    data_augmentation,

    # --------------------------------------------------------
    # BLOCK 1
    # --------------------------------------------------------

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # BLOCK 2
    # --------------------------------------------------------

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # BLOCK 3
    # --------------------------------------------------------

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # BLOCK 4
    # --------------------------------------------------------

    layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------------

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(
        0.4
    ),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n========================================")
print("MODEL SUMMARY")
print("========================================")

model.summary()


summary_path = os.path.join(
    RESULT_DIR,
    "model_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    model.summary(
        print_fn=lambda x:
        f.write(x + "\n")
    )


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# TRAINING
# ============================================================

print("\n========================================")
print("TRAINING")
print("========================================")

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    shuffle=True,

    verbose=1
)


# ============================================================
# TRAINING HISTORY CSV
# ============================================================

history_csv = os.path.join(
    RESULT_DIR,
    "training_history.csv"
)

with open(
    history_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Epoch",
        "Training_Accuracy",
        "Validation_Accuracy",
        "Training_Loss",
        "Validation_Loss"
    ])

    for i in range(EPOCHS):

        writer.writerow([

            i + 1,

            history.history["accuracy"][i],

            history.history["val_accuracy"][i],

            history.history["loss"][i],

            history.history["val_loss"][i]
        ])


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Training vs Validation Accuracy"
)

plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "accuracy.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training"
)

plt.plot(
    history.history["val_loss"],
    label="Validation"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training vs Validation Loss"
)

plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "loss.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# VALIDATION
# ============================================================

print("\n========================================")
print("VALIDATION")
print("========================================")

loss, accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0
)

probabilities = model.predict(
    X_val,
    verbose=0
).ravel()

y_pred = (
    probabilities >= 0.5
).astype(int)


print(
    f"Validation Loss     : {loss:.4f}"
)

print(
    f"Validation Accuracy : {accuracy * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_val,
    y_pred,
    labels=[0, 1]
)

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)


plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.xticks(
    [0, 1],
    ["Not Yawning", "Yawning"]
)

plt.yticks(
    [0, 1],
    ["Not Yawning", "Yawning"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "Confusion Matrix"
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=16
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_val,
    y_pred,

    labels=[0, 1],

    target_names=[
        "Not Yawning",
        "Yawning"
    ],

    zero_division=0
)

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(report)


with open(
    os.path.join(
        RESULT_DIR,
        "classification_report.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(report)


# ============================================================
# PRECISION / RECALL / F1
# ============================================================

precision = precision_score(
    y_val,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_val,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_val,
    y_pred,
    zero_division=0
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1-Score  : {f1 * 100:.2f}%"
)


values = [
    precision,
    recall,
    f1
]

names = [
    "Precision",
    "Recall",
    "F1-Score"
]


plt.figure(figsize=(7, 5))

plt.bar(
    names,
    values
)

plt.ylim(
    0,
    1
)

plt.ylabel(
    "Score"
)

plt.title(
    "Precision / Recall / F1-Score"
)

plt.grid(
    axis="y"
)

for i, v in enumerate(values):

    plt.text(
        i,
        v + 0.02,
        f"{v:.2f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "precision_recall_f1.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# ROC / AUC
# ============================================================

auc_score = None

if len(np.unique(y_val)) == 2:

    fpr, tpr, _ = roc_curve(
        y_val,
        probabilities
    )

    auc_score = roc_auc_score(
        y_val,
        probabilities
    )

    print(
        f"\nROC AUC : {auc_score:.4f}"
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {auc_score:.3f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        label="Random"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            GRAPH_DIR,
            "roc_curve.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# PERFORMANCE FILE
# ============================================================

performance_path = os.path.join(
    RESULT_DIR,
    "performance_metrics.txt"
)

with open(
    performance_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CNN YAWNING DETECTION PERFORMANCE\n\n"
    )

    f.write(
        f"Validation Loss     : {loss:.4f}\n"
    )

    f.write(
        f"Validation Accuracy : {accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Precision           : {precision * 100:.2f}%\n"
    )

    f.write(
        f"Recall              : {recall * 100:.2f}%\n"
    )

    f.write(
        f"F1-Score            : {f1 * 100:.2f}%\n"
    )

    if auc_score is not None:

        f.write(
            f"ROC AUC             : {auc_score:.4f}\n"
        )

    f.write(
        "\nConfusion Matrix:\n"
    )

    f.write(
        str(cm)
    )


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    MODEL_PATH
)

print("\n========================================")
print("MODEL SAVED")
print("========================================")

print(
    MODEL_PATH
)


# ============================================================
# TEST CSV
# ============================================================

test_csv = os.path.join(
    RESULT_DIR,
    "test_results.csv"
)

if not os.path.exists(test_csv):

    with open(
        test_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        csv.writer(f).writerow([

            "Image",
            "Prediction",
            "Confidence",
            "Mouth_Width",
            "Mouth_Height",
            "MAR",
            "Yawning_Level"

        ])


# ============================================================
# IMAGE TESTING
# ============================================================

while True:

    answer = input(
        "\nDo you want to test an image? (yes/no): "
    ).lower().strip()

    if answer != "yes":
        break

    image_path = input(
        "Enter image path: "
    ).strip().strip('"')

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "ERROR: Image not found!"
        )

        continue

    display = image.copy()


    # ========================================================
    # FACE DETECTION
    # ========================================================

    face, face_box = detect_and_crop_face(
        image
    )


    if face is None:

        print(
            "\nFace not detected!"
        )

        print(
            "CNN prediction was not performed."
        )

        cv2.putText(
            display,
            "FACE NOT DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        name, ext = os.path.splitext(
            os.path.basename(image_path)
        )

        output = os.path.join(
            RESULT_DIR,
            name + "_result" + ext
        )

        cv2.imwrite(
            output,
            display
        )

        cv2.imshow(
            "Yawning Detection",
            display
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        continue


    # ========================================================
    # FACE BOX
    # ========================================================

    x, yf, w, h = face_box

    cv2.rectangle(
        display,
        (x, yf),
        (x + w, yf + h),
        (255, 0, 0),
        2
    )


    # ========================================================
    # CNN INPUT
    # ========================================================

    test = cv2.resize(
        face,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    test = cv2.cvtColor(
        test,
        cv2.COLOR_BGR2RGB
    )

    test = (
        test.astype(np.float32) / 255.0
    )

    test = np.expand_dims(
        test,
        axis=0
    )


    # ========================================================
    # CNN PREDICTION
    # ========================================================

    p = model.predict(
        test,
        verbose=0
    )[0][0]


    if p >= 0.5:

        status = "YAWNING"

        confidence = p * 100

    else:

        status = "NOT YAWNING"

        confidence = (1 - p) * 100


    # ========================================================
    # MOUTH ANALYSIS
    # ========================================================

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    mw = 0
    mh = 0
    MAR = 0
    level = "UNKNOWN"


    # --------------------------------------------------------
    # MOUTH REGION
    # --------------------------------------------------------

    mx1 = max(
        0,
        int(x + 0.20 * w)
    )

    mx2 = min(
        image.shape[1],
        int(x + 0.80 * w)
    )

    my1 = max(
        0,
        int(yf + 0.55 * h)
    )

    my2 = min(
        image.shape[0],
        int(yf + 0.90 * h)
    )


    mouth = gray[
        my1:my2,
        mx1:mx2
    ]


    # ========================================================
    # MAR CALCULATION
    # ========================================================

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

            c = max(
                contours,
                key=cv2.contourArea
            )

            area = cv2.contourArea(c)

            if area > 20:

                _, _, mw, mh = cv2.boundingRect(c)

                if mw > 0:

                    MAR = mh / mw


    # ========================================================
    # YAWNING LEVEL
    # ========================================================

    if MAR >= 0.45:

        level = "HIGH"

    elif MAR >= 0.25:

        level = "MEDIUM"

    else:

        level = "LOW"


    # ========================================================
    # MOUTH REGION BOX
    # ========================================================

    cv2.rectangle(
        display,
        (mx1, my1),
        (mx2, my2),
        (0, 255, 0),
        2
    )


    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "YAWNING DETECTION RESULT"
    )

    print(
        "========================================"
    )

    print(
        "Prediction   :",
        status
    )

    print(
        f"Confidence   : {confidence:.2f}%"
    )

    print(
        f"Mouth Width  : {mw:.1f}px"
    )

    print(
        f"Mouth Height : {mh:.1f}px"
    )

    print(
        f"MAR          : {MAR:.3f}"
    )

    print(
        "Yawning Level:",
        level
    )


    # ========================================================
    # DISPLAY TEXT
    # ========================================================

    text_lines = [

        f"Status: {status}",

        f"Confidence: {confidence:.1f}%",

        f"Mouth W: {mw:.1f}px",

        f"Mouth H: {mh:.1f}px",

        f"MAR: {MAR:.3f}",

        f"Yawning Level: {level}"

    ]


    for i, line in enumerate(
        text_lines
    ):

        cv2.putText(

            display,

            line,

            (20, 35 + i * 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 255),

            2
        )


    # ========================================================
    # SAVE RESULT IMAGE
    # ========================================================

    name, ext = os.path.splitext(
        os.path.basename(image_path)
    )

    output = os.path.join(
        RESULT_DIR,
        name + "_result" + ext
    )

    cv2.imwrite(
        output,
        display
    )


    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(
        test_csv,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        csv.writer(f).writerow([

            image_path,

            status,

            f"{confidence:.2f}",

            f"{mw:.1f}",

            f"{mh:.1f}",

            f"{MAR:.3f}",

            level

        ])


    print(
        "\nResult saved:",
        output
    )


    # ========================================================
    # SHOW IMAGE
    # ========================================================

    cv2.imshow(
        "Yawning Detection",
        display
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# ============================================================
# COMPLETED
# ============================================================

print("\n========================================")
print("CNN YAWNING DETECTION SYSTEM COMPLETED")
print("========================================")

print(
    "Model  :",
    MODEL_PATH
)

print(
    "Results:",
    RESULT_DIR
)

print(
    "Graphs :",
    GRAPH_DIR
)

