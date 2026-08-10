import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download
import groq
from reportlab.pdfgen import canvas
import os

# -------------------------------
# Descargar modelo desde HuggingFace Hub
# -------------------------------
model_path = hf_hub_download(
    repo_id="TU_USUARIO/coffee-leaf-model",   # Reemplaza con tu usuario
    filename="coffee_leaf_model.h5"
)

model = tf.keras.models.load_model(model_path)

# -------------------------------
# Configuración de Groq API
# -------------------------------
client = groq.Client(api_key="TU_API_KEY")  # Reemplaza con tu API Key

def groq_api_call(disease):
    prompt = f"Genera descripción, recomendaciones técnicas, buenas prácticas y acciones de seguimiento para la enfermedad {disease} en hojas de café."
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role":"user","content":prompt}]
    )
    return response.choices[0].message.content

# -------------------------------
# Función de predicción
# -------------------------------
def predict_leaf(image_path):
    img = Image.open(image_path).resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    label = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return label, confidence

# -------------------------------
# Generar PDF con ReportLab
# -------------------------------
def generar_pdf(disease, confidence, recomendaciones):
    filename = "diagnostico.pdf"
    c = canvas.Canvas(filename)
    c.drawString(100, 750, "Diagnóstico de Hoja de Café")
    c.drawString(100, 700, f"Enfermedad detectada: {disease}")
    c.drawString(100, 680, f"Confianza: {confidence:.2f}%")
    c.drawString(100, 640, "Recomendaciones:")
    c.drawString(120, 620, recomendaciones[:200])  # corta texto largo
    c.save()
    return filename

# -------------------------------
# Interfaz Streamlit
# -------------------------------
st.title("🌱 Coffee Leaf Disease Detection App")

uploaded_file = st.file_uploader("Sube una imagen de la hoja de café", type=["jpg","png"])

if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    label, confidence = predict_leaf("temp.jpg")

    st.write(f"✅ Enfermedad detectada: {label}")
    st.write(f"📊 Confianza: {confidence:.2f}%")

    recomendaciones = groq_api_call(label)
    st.subheader("📖 Recomendaciones generadas por IA")
    st.write(recomendaciones)

    if st.button("Generar PDF"):
        pdf_file = generar_pdf(label, confidence, recomendaciones)
        with open(pdf_file, "rb") as f:
            st.download_button("Descargar PDF", f, file_name="diagnostico.pdf")
