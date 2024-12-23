import os
import numpy as np
import cv2
import tensorflow as tf
import ssl 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS  # Enable CORS
from tensorflow.keras.applications import MobileNetV2

ssl._create_default_https_context = ssl._create_unverified_context

model_dasar = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Flask app and CORS setup
app = Flask(__name__)
CORS(app, resources={r"/classify": {"origins": "*"}})  # Mengizinkan semua domain # Set your frontend URL for allowed origins

# 1. Direktori Dataset
base_dir = '/Users/dewaputufajarw/Downloads/Project PCD'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Direktori tidak ditemukan: {train_dir}")
if not os.path.exists(valid_dir):
    raise FileNotFoundError(f"Direktori tidak ditemukan: {valid_dir}")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Direktori tidak ditemukan: {test_dir}")

# 2. Pra-pemrosesan Data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(160, 160),  # Sesuaikan ukuran gambar sesuai dengan MobileNetV2
    batch_size=32,
    class_mode='categorical')

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(160, 160),  # Sesuaikan ukuran gambar sesuai dengan MobileNetV2
    batch_size=32,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(160, 160),  # Sesuaikan ukuran gambar sesuai dengan MobileNetV2
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

# 3. Load MobileNetV2 tanpa fully connected layer (include_top=False)
mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3))

# Freeze semua lapisan MobileNetV2
for layer in mobilenet.layers:
    layer.trainable = False

# Membangun model
model = Sequential([
    mobilenet,
    GlobalAveragePooling2D(),  # Pooling global
    Dense(len(train_generator.class_indices), activation='softmax')  # Klasifikasi multiclass
])

# Kompilasi model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Definisi callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('model_terbaik.keras', save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]

# Pelatihan Model
model.fit(
    train_generator,
    epochs=20,
    validation_data=valid_generator,
    callbacks=callbacks
)

# Flask API
@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Ensure the filename is safe
    filename = secure_filename(file.filename)
    file_path = os.path.join('temp', filename)
    os.makedirs('temp', exist_ok=True)
    file.save(file_path)

    # Pra-pemrosesan gambar
    image = cv2.imread(file_path)
    image = cv2.resize(image, (160, 160))  # Sesuaikan ukuran gambar dengan model
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Prediksi menggunakan model
    prediction = model.predict(image)
    label = np.argmax(prediction, axis=1)[0]

    # Map label ke kelas
    class_indices = train_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    result_label = class_names[label]

    # Hapus file sementara
    os.remove(file_path)

    return jsonify({"label": result_label})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)

# Define function to extract SIFT features
def extract_sift_features(image, image_size=(224, 224), fixed_length=128):
    img_resized = cv2.resize(image, image_size)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_resized, None)

    # If no descriptors found, return an empty array
    if descriptors is None:
        descriptors = np.zeros((1, fixed_length))

    # Ensure descriptors have the same length
    if descriptors.shape[0] > 1:
        descriptors = np.mean(descriptors, axis=0).reshape(1, -1)

    # If descriptors are too short, pad with zeros
    if descriptors.shape[1] < fixed_length:
        descriptors = np.pad(descriptors, ((0, 0), (0, fixed_length - descriptors.shape[1])), mode='constant')

    return descriptors.flatten()  # Flatten to 1D array
# Load dataset function (loading images from a directory)
def load_images(image_dir, use_hog=True):
    data = []
    labels = []

    for label in os.listdir(image_dir):
        class_dir = os.path.join(image_dir, label)
        if os.path.isdir(class_dir):
            for image_path in glob.glob(os.path.join(class_dir, "*.jpg")):
                img = cv2.imread(image_path)

                # Apply either SIFT based on the parameter
                                   features = extract_sift_features(img)

                data.append(features)
                labels.append(label)

    return np.array(data), np.array(labels)


# Load and evaluate using SIFT features
print("\nEvaluating SIFT Features:")

# Load SIFT features for train, validation, and test sets
X_train_sift, y_train_sift = load_images(train_dir, use_hog=False)
X_valid_sift, y_valid_sift = load_images(valid_dir, use_hog=False)
X_test_sift, y_test_sift = load_images(test_dir, use_hog=False)

# Combine train and validation data for final training
X_train_combined_sift = np.concatenate((X_train_sift, X_valid_sift), axis=0)
y_train_combined_sift = np.concatenate((y_train_sift, y_valid_sift), axis=0)

# Classify and evaluate SIFT features
classify_and_evaluate(X_train_combined_sift, X_test_sift, y_train_combined_sift, y_test_sift)
