import os
import numpy as np
import cv2
import tensorflow as tf
import ssl 
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
CORS(app, resources={r"/classify": {"origins": "http://localhost:5500"}})  # Set your frontend URL for allowed origins

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
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

# 3. Ekstraksi Fitur dengan SIFT
def ekstrak_fitur_sift(jalur_gambar):
    gambar = cv2.imread(jalur_gambar, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gambar, None)
    return descriptors


# Mengumpulkan fitur SIFT dari dataset pelatihan
fitur_sift = []
label = []
for label_kelas, direktori_kelas in enumerate(os.listdir(train_dir)):
    jalur_kelas = os.path.join(train_dir, direktori_kelas)
    
    # Pastikan hanya memproses direktori, bukan file seperti .DS_Store
    if os.path.isdir(jalur_kelas):
        for file_gambar in os.listdir(jalur_kelas):
            jalur_gambar = os.path.join(jalur_kelas, file_gambar)
            
            # Pastikan hanya memproses file gambar (misalnya dengan ekstensi .jpg atau .png)
            if file_gambar.lower().endswith(('.png', '.jpg', '.jpeg')):
                descriptors = ekstrak_fitur_sift(jalur_gambar)
                if descriptors is not None:
                    fitur_sift.append(descriptors)
                    label.append(label_kelas)
       

# Menggabungkan fitur SIFT menjadi satu matriks fitur
fitur_sift_tergabung = np.vstack(fitur_sift)

# 4. CNN Pretrained (MobileNetV2)
model_dasar = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model_dasar.trainable = False

def ekstrak_fitur_cnn(generator, model_dasar):
    fitur = []
    for batch, _ in generator:
        batch_fitur = model_dasar.predict(batch)
        fitur.append(batch_fitur)
    return np.vstack(fitur)

x = model_dasar.output
x = GlobalAveragePooling2D()(x)
fitur_cnn = Dense(256, activation='relu')(x)
fitur_cnn = Dropout(0.5)(fitur_cnn)

# 5. Penggabungan Fitur
input_sift = tf.keras.Input(shape=(fitur_sift_tergabung.shape[1],))
fitur_tergabung = Concatenate()([fitur_cnn, input_sift])

fitur_cnn = ekstrak_fitur_cnn(train_generator, model_dasar)
fitur_tergabung = np.hstack([fitur_cnn, fitur_sift_tergabung])  # Gabungkan fitur CNN dan SIFT


# 6. Pelatihan dan Optimasi
output = Dense(len(train_generator.class_indices), activation='softmax')(fitur_tergabung)
model = Model(inputs=[model_dasar.input, input_sift], outputs=output)

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
    x=fitur_tergabung,
    y=label,  # Label dalam bentuk array NumPy
    validation_data=(valid_fitur, valid_label),  # Lakukan hal yang sama untuk validasi
    epochs=10,
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
    image = cv2.resize(image, (224, 224))
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
