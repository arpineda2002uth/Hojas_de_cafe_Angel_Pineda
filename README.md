# Hojas_de_cafe_Angel_Pineda

Hola ingeniera, lamentablemente no pude terminar la tarea, al momento de subirlo al repositorio de github y montarlo en streamlit me seguía tirando siempre el mismo error, estuve desde ayer en la noche y parte del día del hoy intentando solucionar cada error, espero poder ser evaluado con el código, una disculpa y buenas noches

## 📖 Descripción
Este proyecto es una aplicación web desarrollada con Streamlit que permite detectar enfermedades en hojas de café usando un modelo de entrenado con TensorFlow.  
El sistema analiza una imagen de la hoja, predice la enfermedad, muestra el nivel de confianza y genera recomendaciones técnicas con ayuda de Groq API. Además, permite exportar un informe en PDF con ReportLab.

## 🛠️ Tecnologías utilizadas
- Streamlit → interfaz web interactiva.  
- TensorFlow → modelo de clasificación de imágenes.  
- NumPy → manejo de arrays y cálculos.  
- Pillow → carga y visualización de imágenes.  
- Groq → generación de descripciones y recomendaciones con IA.  
- ReportLab → creación de reportes PDF.  

## 🚀 Cómo usar la aplicación
1. Abre el sitio web desplegado en Streamlit Cloud.  
2. Sube una imagen de una hoja de café en formato `.jpg` o `.png`.  
3. El sistema mostrará:  
   - La enfermedad detectada.  
   - El porcentaje de confianza del modelo.  
   - Recomendaciones técnicas generadas con IA.  
4. (Opcional) Haz clic en Generar PDF para descargar un informe con el diagnóstico.  

## 📂 Estructura del proyecto
```
📁## Hojas de Café
 ┣ 📄 app.py                # Código principal de la aplicación
 ┣ 📄 requirements.txt      # Dependencias necesarias
 ┗ 📄 README.md             # Documentación del proyecto
```
## 🌐 Despliegue
El proyecto está desplegado en Streamlit Community Cloud.  
👉 Enlace: [Coffee Leaf Disease App](https://streamlit.io/cloud) 

## ✨ Autor
Proyecto desarrollado por Angel Ricardo Pineda Diaz como parte de la asignatura Computación en la Nube en la Universidad Tecnológica de Honduras (UTH).
