import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO

# Configuración de la página
st.set_page_config(page_title="Detección de Objetos", layout="wide")

# Título de la aplicación
st.title("Detección de Objetos con YOLOv8")

# Cargar el modelo (asegúrate de tener best.pt en el directorio correcto)
@st.cache_resource
def load_model():
    try:
        model = YOLO("/model/best.pt")
        return model
    except Exception as e:
        st.error(f"No se pudo cargar el modelo: {e}")
        return None

model = load_model()

# Opciones para el usuario
option = st.radio("Seleccione una opción:", ("Subir imagen", "Usar cámara"))

if option == "Subir imagen":
    uploaded_file = st.file_uploader("Elija una imagen...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Leer la imagen
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_column_width=True)
        
        # Convertir a formato OpenCV
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        
        # Realizar predicción
        if model is not None:
            results = model.predict(image_cv, imgsz=640)
            
            # Mostrar resultados
            for r in results:
                im_array = r.plot()  # imagen con las detecciones
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                st.image(im, caption="Resultado de la detección", use_column_width=True)

else:  # Usar cámara
    picture = st.camera_input("Tome una foto")
    
    if picture:
        # Leer la imagen de la cámara
        image = Image.open(picture)
        
        # Convertir a formato OpenCV
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        
        # Realizar predicción
        if model is not None:
            results = model.predict(image_cv, imgsz=640)
            
            # Mostrar resultados
            for r in results:
                im_array = r.plot()  # imagen con las detecciones
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                st.image(im, caption="Resultado de la detección", use_column_width=True)

# Notas adicionales
st.sidebar.markdown("""
### Instrucciones:
1. Seleccione "Subir imagen" para cargar una foto desde su dispositivo
2. O seleccione "Usar cámara" para tomar una foto con su teléfono
3. Espere a que el modelo procese la imagen y muestre los resultados
""")