import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# --- 1. LOAD DATASET ---
# Mengambil dataset Fashion MNIST yang sudah disediakan oleh TensorFlow/Keras
mnist = tf.keras.datasets.fashion_mnist
# Membagi data menjadi 2 bagian: Training (untuk belajar) dan Testing (untuk ujian akhir)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# --- 2. PREPROCESSING (NORMALISASI) ---
# Gambar asli memiliki nilai pixel 0-255. 
# Kita ubah menjadi skala 0.0 - 1.0 agar proses perhitungan matematika di dalam AI lebih cepat dan akurat.
x_train, x_test = x_train / 255.0, x_test / 255.0

# --- 3. DEFINISI LABEL ---
# Karena output dataset berupa angka (0-9), kita buat daftar nama agar kita tahu arti angka tersebut
class_names = [
    'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

label_unik = np.unique(y_train)
print("Angka label yang ada di dataset ini adalah:")
print(label_unik)

# --- 4. MEMBANGUN ARSITEKTUR MODEL (OTAK AI) ---
model = tf.keras.models.Sequential([
    # Flatten: Mengubah gambar 2D (28x28 pixel) menjadi satu baris panjang (784 pixel)
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    
    # Dense (Hidden Layer): Lapisan saraf dengan 512 neuron. 
    # 'relu' adalah fungsi aktivasi yang membantu AI mempelajari pola yang kompleks.
    tf.keras.layers.Dense(512, activation='relu'),
    
    # Dense (Output Layer): Lapisan terakhir dengan 10 neuron (sesuai jumlah kategori pakaian).
    # 'softmax' mengubah hasil menjadi probabilitas (misal: 90% yakin ini sepatu).
    tf.keras.layers.Dense(10, activation='softmax')
])

# --- 5. KONFIGURASI MODEL (COMPILE) ---
model.compile(
    optimizer='adam', # Algoritma pelatih yang menyesuaikan model agar semakin pintar
    loss='sparse_categorical_crossentropy', # Cara menghitung seberapa besar kesalahan AI
    metrics=['accuracy'] # Kita ingin melihat seberapa tinggi tingkat akurasi (persentase benar)
)

# --- 6. PROSES BELAJAR (TRAINING) ---
# AI belajar dengan membaca data training sebanyak 5 kali (epochs=5)
print("\n--- Memulai Proses Belajar ---")
model.fit(x_train, y_train, epochs=5)

# --- 7. EVALUASI (UJIAN AKHIR) ---
# Menguji kemampuan AI menggunakan data testing yang belum pernah ia lihat saat belajar
print("\n--- Evaluasi dengan Data Ujian ---")
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Akurasi Akhir: {accuracy * 100:.2f}%")

# --- 8. PREDIKSI DAN VISUALISASI ---
# Mencoba menebak seluruh gambar di data testing
predictions = model.predict(x_test)

# Mengambil satu contoh gambar (index ke-4) untuk ditampilkan hasilnya
index_gambar = 4
plt.figure(figsize=(6,3))
plt.imshow(x_test[index_gambar], cmap='gray')

# np.argmax mengambil angka tertinggi dari hasil prediksi (kategori yang paling diyakini AI)
nama_prediksi = class_names[np.argmax(predictions[index_gambar])]
nama_asli = class_names[y_test[index_gambar]]

plt.title(f"Prediksi: {nama_prediksi} (Asli: {nama_asli})")
plt.axis('off')
plt.savefig('hasil_prediksi.png') # Menyimpan hasil visualisasi
print(f"\nHasil Prediksi Gambar {index_gambar}: {nama_prediksi}")
print(f"Jawaban Sebenarnya: {nama_asli}")