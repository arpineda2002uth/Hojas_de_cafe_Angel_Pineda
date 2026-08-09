import gdown

# ID del archivo en Google Drive
file_id = "TU_ID_DE_DRIVE"
url = f"https://drive.google.com/file/d/1S_l7IrR3zy0FrqH9nb13xsvoXsOTyr9b/view?usp=drive_link"
output = "coffee_leaf_model.h5"

# Descargar modelo desde Drive
gdown.download(url, output, quiet=False)

# Cargar modelo
model = tf.keras.models.load_model(output)

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import groq
from reportlab.pdfgen import canvas

# Cargar modelo entrenado
model = tf.keras.models.load_model("coffee_leaf_model.h5")

# Definir clases (ajusta según tu dataset)
classes = ["Healthy", "Rust", "Mineral Deficiency"]

# Función de predicción
def predict_leaf(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    confidence = np.max(prediction) * 100
    label = classes[np.argmax(prediction)]
    return label, confidence

# Conexión con Groq API
client = groq.Client(api_key="TU_API_KEY")  # Reemplaza con tu API Key real

def groq_api_call(disease):
    prompt = f"Genera descripción, recomendaciones técnicas, buenas prácticas y acciones de seguimiento para la enfermedad {disease} en hojas de café."
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role":"user","content":prompt}]
    )
    return response.choices[0].message.content

# Generar PDF con ReportLab
def generar_pdf(disease, confidence, recomendaciones):
    c = canvas.Canvas("diagnostico.pdf")
    c.drawString(100, 750, f"Diagnóstico: {disease}")
    c.drawString(100, 730, f"Confianza: {confidence:.2f}%")
    c.drawString(100, 710, "Recomendaciones:")
    c.drawString(120, 690, recomendaciones)
    c.save()
    return "diagnostico.pdf"

# Interfaz Streamlit
def main():
    st.title("🌱 Detección de Enfermedades en Hojas de Café")

    uploaded_file = st.file_uploader("Sube una imagen de la hoja", type=["jpg","png","jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Imagen cargada", use_column_width=True)

        img.save("temp.jpg")
        label, confidence = predict_leaf("temp.jpg")

        st.write(f"✅ Enfermedad detectada: {label}")
        st.write(f"📊 Confianza: {confidence:.2f}%")

        recomendaciones = groq_api_call(label)
        st.subheader("📖 Recomendaciones generadas por IA")
        st.write(recomendaciones)

        if st.button("Generar PDF"):
            pdf_path = generar_pdf(label, confidence, recomendaciones)
            st.success(f"PDF generado: {pdf_path}")

if __name__ == "__main__":
    main()
