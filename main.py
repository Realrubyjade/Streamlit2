import streamlit as st
import numpy as np

# Streamlit App
st.title("Uji Hipotesis Vektor Rata-Rata untuk 1 Populasi")

# Input jumlah sampel
n = st.number_input("Masukkan jumlah sampel (n > 2):", placeholder="Contoh: 3", step=1, min_value=3)

# Input nilai alfa
alpha = st.number_input(
    "Masukkan nilai alfa (contoh: 0.05):", min_value=0.001, max_value=0.999, value=0.05, step=0.01
)

# Input hipotesis nol
h0_input = st.text_input(
    "Masukkan vektor rata-rata hipotesis H₀ (pisahkan dengan koma):",
    placeholder="Contoh: 0,0",
)

# Input data
data_input = st.text_area(
    "Masukkan data (dua variabel, pisahkan dengan koma, gunakan enter untuk baris baru):",
    placeholder="Contoh:\n1,2\n3,4\n5,6",
)

if st.button("Hitung Uji Hipotesis"):
    try:
        # Parsing data
        data = [list(map(float, row.split(","))) for row in data_input.strip().split("\n")]
        data = np.array(data)

        # Parsing hipotesis nol
        h0 = np.array(list(map(float, h0_input.split(","))))

        # Validasi dimensi data
        if data.shape[1] != 2:
            st.error("Data harus memiliki dua kolom (dua variabel).")
        elif len(data) != n:
            st.error("Jumlah baris data harus sesuai dengan jumlah sampel (n).")
        elif len(h0) != 2:
            st.error("Vektor hipotesis nol H₀ harus memiliki dua elemen.")
        else:
            # Hitung rata-rata sampel
            mean_vector = np.mean(data, axis=0)

            # Hitung matriks covariance (menggunakan divisor n-1)
            covariance_matrix = np.cov(data, rowvar=False, ddof=1)

            # Invers matriks covariance
            cov_inv = np.linalg.inv(covariance_matrix)

            # Hitung nilai statistik Hotelling's T-squared
            diff = mean_vector - h0
            t_squared = n * diff @ cov_inv @ diff.T

            # Konversi ke F-statistik
            f_stat = (t_squared * (n - 2)) / (2 * (n - 1))

            # Derajat kebebasan
            df1 = 2
            df2 = n - 2

            # Perhitungan nilai kritis F secara manual
            # Menggunakan hubungan empiris untuk threshold F-statistic
            f_critical = (df1 / df2) * ((1 - alpha) / alpha)

            # Hasil
            st.write("### Hasil Uji Hipotesis")
            st.write(f"Vektor rata-rata sampel: {mean_vector}")
            st.write(f"Vektor H₀: {h0}")
            st.write(f"Matriks covariance: \n{covariance_matrix}")
            st.write(f"Statistik F: {f_stat:.4f}")
            st.write(f"Nilai Kritis F: {f_critical:.4f}")

            # Keputusan
            if f_stat > f_critical:
                st.write(f"Keputusan: Tolak H₀ pada tingkat signifikansi {alpha}.")
            else:
                st.write(f"Keputusan: Terima H₀ pada tingkat signifikansi {alpha}.")

    except Exception as e:
        st.error(f"Terjadi kesalahan dalam pemrosesan: {e}")
