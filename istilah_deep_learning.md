# 📘 Kamus Sederhana Deep Learning (Fashion MNIST)

Dokumen ini adalah panduan istilah untuk membantumu memahami setiap bagian dari proyek AI ini tanpa bahasa teknis yang rumit.

---

### 📦 1. Data & Persiapan
*   **Dataset**: Kumpulan besar data (gambar & label) yang digunakan untuk mengajar AI.
*   **Training Set (`x_train`, `y_train`)**: Buku pelajaran. Data yang dilihat AI berkali-kali untuk belajar pola.
*   **Testing Set (`x_test`, `y_test`)**: Lembar ujian. Data baru yang belum pernah dilihat AI, digunakan untuk mengetes kepintarannya.
*   **Label**: Kunci jawaban. Angka (0-9) yang memberi tahu komputer nama dari gambar tersebut.
*   **Normalization**: Proses mengecilkan angka (dari 0-255 jadi 0-1) agar otak AI tidak "pusing" saat menghitung angka raksasa.

### 🧠 2. Struktur Otak AI (Arsitektur)
*   **Model**: Sebutan untuk "Otak Digital" yang sedang kita bangun.
*   **Layer (Lapisan)**: Tahapan pemrosesan di dalam otak AI. Data masuk lewat satu pintu, diproses, lalu dilempar ke pintu berikutnya.
*   **Neuron (Saraf)**: Unit terkecil di dalam Layer. Tugasnya cuma satu: memberikan skor (0 sampai 1) pada sebuah informasi.
*   **Flatten**: Proses "meratakan" gambar dari bentuk kotak (2D) menjadi satu baris panjang agar bisa dibaca oleh saraf.
*   **Dense**: Jenis lapisan di mana semua saraf saling terhubung satu sama lain (seperti jaringan laba-laba).
*   **Activation Function**: Aturan main di setiap saraf. 
    *   **ReLU**: Membuang informasi yang tidak berguna (negatif) dan hanya meneruskan yang penting.
    *   **Softmax**: Mengubah hasil akhir menjadi persentase (misal: 95% yakin ini Sepatu).

### 🎓 3. Proses Belajar
*   **Weights (Bobot)**: Kekuatan hubungan antar saraf. AI belajar dengan mengubah-ubah angka bobot ini sampai tebakannya benar.
*   **Epochs**: Jumlah putaran belajar. 1 Epoch = AI sudah membaca seluruh buku pelajaran sebanyak satu kali.
*   **Optimizer (Adam)**: Pelatih/Coach. Dialah yang bertugas memperbaiki angka-angka di dalam otak AI setiap kali AI salah menebak.
*   **Loss Function**: Alat ukur kesalahan. Semakin besar angka Loss, artinya tebakan AI semakin ngawur. Tujuan kita adalah membuat Loss sekecil mungkin.

### 📊 4. Hasil & Penilaian
*   **Accuracy**: Nilai rapot. Seberapa sering AI menebak dengan benar (misal: 0.85 = 85% benar).
*   **Predict**: Saat kita menyuruh AI yang sudah pintar untuk menebak sebuah gambar baru.
*   **np.argmax**: Perintah untuk mencari "siapa pemenang skor tertinggi" dari hasil prediksi AI.

---
> **Tips:** Simpan file ini sebagai referensi setiap kali kamu membaca kode di `main.py`!
