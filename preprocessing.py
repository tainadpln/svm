import cv2
import numpy as np
import os

def load_data(path):
    X = []
    y = []

    categories = {
        "NORMAL": 1,
        "PNEUMONIA": -1
    }

    for category, label in categories.items():
        folder_path = os.path.join(path, category)

        for image_name in os.listdir(os.path.join(folder_path)):
            if image_name.lower().endswith((".jpeg", ".jpg")):
                image_path = os.path.join(folder_path, image_name)
                img = cv2.imread(image_path)
                if img is None:
                    continue

                img_resized = cv2.resize(img, (128, 128))
                X.append(img_resized)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    X = X.reshape(X.shape[0], -1).astype("float32") / 255.0

    return X, y
