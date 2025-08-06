import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.efficientnet_v2 import preprocess_input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Aplikasi Klasifikasi Jenis Pisang",
    page_icon="🍌",
    layout="wide"
)

# --- Fungsi untuk Memuat Model ---
@st.cache_resource
def load_keras_model():
    model_path = 'versi3.keras'  # Pastikan file ini ada di direktori kerja
    model = load_model(model_path)
    return model

model = load_keras_model()

# --- Daftar Kelas ---
banana_classes = [
    'Pisang Ambon', 'Pisang Cavendish', 'Pisang Genderuwo', 'Pisang Kepok', 'Pisang Tanduk'
]

# --- Fungsi Prediksi ---
def predict_banana_type(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224)).convert("RGB")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # ✅ Sesuai dengan training
    predictions = model.predict(img_array)[0]
    return predictions

# --- Sidebar ---
with st.sidebar:
    st.image("banaclass.png", width=100)
    st.title("BanaClass")
    menu = st.selectbox("Pilih Halaman", ("Home", "Jenis Pisang", "About"))

# --- Halaman HOME ---
if menu == "Home":
    st.markdown("""
        <style>
        .big-title {
            font-size: 3em;
            color: #4CAF50;
            text-align: center;
            margin-bottom: 10px;
            font-family: 'Segoe UI', sans-serif;
            text-shadow: 1px 1px 3px #aaa;
        }
        .welcome-message {
            font-size: 1.2em;
            color: #555;
            text-align: center;
            margin-bottom: 30px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='big-title'>🍌 Klasifikasi Jenis Pisang 🍌</h1>", unsafe_allow_html=True)
    st.markdown("<p class='welcome-message'>Selamat datang di web klasifikasi pisang berbasis Deep Learning EfficientNetV2S</p>", unsafe_allow_html=True)

    st.write("---")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Unggah Gambar Pisang Anda")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            st.image(uploaded_file, caption='Gambar Terunggah', use_container_width=True)
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())

            if st.button("Lakukan Klasifikasi"):
                predictions = predict_banana_type("temp_image.jpg", model)
                
                # --- SISTEM ERROR BARU ---
                confidence_threshold = 0.85 # Tentukan ambang batas keyakinan
                predicted_class_index = np.argmax(predictions)
                confidence = predictions[predicted_class_index]

                if confidence > confidence_threshold:
                    predicted_class = banana_classes[predicted_class_index]
                    st.success(f"**Jenis Pisang Terprediksi:** {predicted_class}")
                    st.info(f"**Keyakinan (Confidence):** {confidence*100:.2f}%")
                    st.session_state["predictions"] = predictions
                else:
                    st.error("Gambar yang diunggah tampaknya **bukan gambar pisang**. Silakan coba unggah gambar pisang.")
                    if "predictions" in st.session_state:
                         del st.session_state["predictions"] # Hapus data prediksi sebelumnya
                # --- AKHIR SISTEM ERROR BARU ---

    with col2:
        if "predictions" in st.session_state:
            st.subheader("Probabilitas Klasifikasi")
            predictions = st.session_state["predictions"]
            df = pd.DataFrame({
                "Jenis Pisang": banana_classes,
                "Probabilitas": predictions
            })
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(df["Jenis Pisang"], df["Probabilitas"], color='gold')
            ax.set_xlabel("Probabilitas")
            ax.set_xlim(0, 1)
            ax.set_title("Distribusi Probabilitas Kelas")
            for i, v in enumerate(df["Probabilitas"]):
                ax.text(v + 0.01, i, f"{v:.2f}", va='center')
            st.pyplot(fig)

            st.write("Tabel Probabilitas:")
            st.dataframe(df.style.format({"Probabilitas": "{:.2%}"}))

# --- Halaman Jenis Pisang ---
elif menu == "Jenis Pisang":
    # CSS Styling untuk Card
    st.markdown("""
    <style>
    .card-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-around;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 12px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.15);
        width: 300px;
        margin: 20px;
        padding: 15px;
        text-align: center;
        transition: transform 0.2s;
    }
    .card:hover {
        transform: scale(1.02);
    }
    .card img {
        width: 100%;
        height: 200px;
        object-fit: cover;
        border-radius: 10px;
    }
    .card-title {
        font-size: 1.2em;
        font-weight: bold;
        margin: 10px 0;
        color: #333;
    }
    .card-desc {
        font-size: 0.9em;
        color: #555;
        text-align: justify;
    }
    </style>
    """, unsafe_allow_html=True)
    # === Judul Halaman ===
    st.title("🍌 Jenis-Jenis Pisang")
    st.write("Berikut adalah informasi tentang beberapa jenis pisang yang bisa dikenali oleh sistem klasifikasi.")
    st.write("---")

    st.set_page_config(page_title="Jenis Pisang", layout="wide")

    # --- Data Jenis Pisang (Lokal)
    banana_info = {
        "Pisang Ambon": {
            "image": "pisang/A.jpg",
            "description": "Pisang Ambon memiliki kulit kuning kehijauan saat matang, daging buah putih kekuningan, dan rasa manis sedikit asam. Pisang ini sangat populer di Indonesia dan sering dimakan langsung sebagai camilan atau digunakan dalam aneka olahan makanan seperti pisang goreng dan kolak. Kandungan vitamin dan mineralnya membuatnya menjadi pilihan sehat bagi banyak orang."
        },
        "Pisang Cavendish": {
            "image": "pisang/cavendish.jpg",
            "description": "Pisang Cavendish adalah jenis pisang yang paling umum secara internasional. Pisang ini dikenal karena bentuknya yang ramping, warna kuning cerah saat matang, dan teksturnya yang lembut. Selain enak dimakan langsung, Cavendish juga sering digunakan dalam smoothies, pancake, dan berbagai kue karena rasanya yang manis."
        },
        "Pisang Genderuwo": {
            "image": "pisang/Genderuwo.jpg",
            "description": "Pisang Genderuwo merupakan varietas lokal yang cukup unik. Pisang ini memiliki kulit berwarna kemerahan atau kehitaman dan bentuk yang sedikit lebih besar. Teksturnya padat dan aroma khasnya membuatnya digemari di daerah tertentu. Biasanya pisang ini diolah terlebih dahulu sebelum dikonsumsi."
        },
        "Pisang Kepok": {
            "image": "pisang/Kepok.jpg",
            "description": "Pisang Kepok memiliki bentuk yang lebih pendek dan gemuk dibanding pisang lainnya. Teksturnya padat, dengan rasa yang tidak terlalu manis. Pisang Kepok banyak digunakan untuk digoreng, dibuat keripik, atau dimasak sebagai campuran dalam makanan tradisional. Pisang ini juga terkenal sebagai pisang diet karena kandungan seratnya."
        },
        "Pisang Tanduk": {
            "image": "pisang/tanduk.jpg",
            "description": "Pisang Tanduk berukuran sangat besar dan berbentuk melengkung seperti tanduk, sesuai dengan namanya. Pisang ini memiliki rasa yang kurang manis saat mentah, sehingga lebih sering dimasak terlebih dahulu. Pisang Tanduk sangat cocok untuk dibuat pisang goreng, kolak, atau dikukus sebagai camilan sehat."
        },
    }

    # --- Style CSS tambahan (justify + card spacing)
    st.markdown("""
        <style>
            .desc-text {
                text-align: justify;
                margin-bottom: 12px;
            }
        </style>
    """, unsafe_allow_html=True)

    # --- Layout per baris
    items_per_row = 2
    items = list(banana_info.items())



    for i in range(0, len(items), items_per_row):
        cols = st.columns(items_per_row)
        for idx, (name, info) in enumerate(items[i:i + items_per_row]):
            with cols[idx]:
                st.image(info["image"], caption=name, width=300)
                st.markdown(
                    f"<div class='desc-text'>{info['description']}</div>",
                    unsafe_allow_html=True
                )
# ===============================
# ======= Halaman About =========
# ===============================
elif menu == "About":
    st.markdown(
        """
        <style>
        .about-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            min-height: 500px;
            text-align: center;
            padding: 50px;
            background-color: #e6ffe6;
            border-radius: 15px;
            box-shadow: 5px 5px 20px rgba(0,0,0,0.15);
            margin-top: 50px;
        }
        .about-title {
           font-size: 3em;
            color: #000000 !important;
            margin-bottom: 20px;
            text-align: center;
            width: 100%;
            display: block;
        }
        .about-author {
            font-size: 1.5em;
            color: #333;
            margin-bottom: 10px;
        }
        .about-description {
            font-size: 1.1em;
            color: #555;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
        }
        .emoji {
            font-size: 2em;
            margin: 0 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="about-container">
            <div style = "text-align: center;">
                <h1 class="about-title">Tentang Aplikasi🚀</h1>
            <p class="about-author">Dibuat oleh <strong>Riski Rahmadan</strong> ✨</p>
            <p class="about-description">
                Aplikasi ini menggunakan model Deep Learning EfficientNetV2S untuk mengklasifikasikan berbagai jenis pisang. 
                Tujuannya adalah memberikan solusi cerdas berbasis gambar yang dapat mengenali jenis pisang secara akurat.
                Aplikasi ini cocok untuk edukasi, pengolahan dataset, dan eksplorasi teknologi AI di bidang pertanian digital.
            </p>
            <p style="margin-top: 30px;">
                <span class="emoji">👨‍💻</span> 
                <span class="emoji">💡</span> 
                <span class="emoji">🌱</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

