import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
import urllib.request
from ultralytics import YOLO

# Configuración de la página
st.set_page_config(page_title="Detección de EPP", layout="wide")

# Título de la aplicación
st.title("Sistema de Detección de Equipo de Protección Personal")

# Configuración del modelo
MODEL_PATH = "model/best.pt"
CLASS_NAMES = {0: "casco", 1: "zapatos", 2: "persona", 3: "chaleco"}

@st.cache_resource
def load_model():
    try:
        # Crear directorio si no existe
        os.makedirs("model", exist_ok=True)
        
        # Descargar el modelo si no existe localmente
        if not os.path.exists(MODEL_PATH):
            with st.spinner('Descargando el modelo... Esto puede tomar unos minutos'):
                urllib.request.urlretrieve(MODEL_PATH)
                st.success("Modelo descargado exitosamente!")
        
        # Cargar el modelo
        model = YOLO(MODEL_PATH)
        return model
    
    except Exception as e:
        st.error(f"No se pudo cargar el modelo: {str(e)}")
        return None

model = load_model()

# Sidebar para selección de EPP requerido
with st.sidebar:
    st.header("Configuración de Detección")
    st.markdown("Seleccione los elementos de protección que desea evaluar:")
    
    # Checkboxes para cada clase
    required_classes = []
    for class_id, class_name in CLASS_NAMES.items():
        if st.checkbox(class_name, value=True, key=f"class_{class_id}"):
            required_classes.append(class_id)
    
    st.markdown("""
    ### Instrucciones:
    1. Seleccione los elementos EPP a evaluar
    2. Suba una imagen o use la cámara
    3. Revise los resultados de detección
    """)

# Opciones para el usuario
option = st.radio("Fuente de imagen:", ("Subir imagen", "Usar cámara"), horizontal=True)

# Contenedor para las imágenes
col1, col2 = st.columns(2)

def process_image(image):
    """Procesa la imagen y devuelve resultados"""
    # Convertir a formato OpenCV
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    # Realizar predicción
    results = model.predict(image_cv, imgsz=640)
    
    # Mostrar todas las detecciones en la imagen analizada
    all_detections = []
    detected_classes = set()
    required_detected = set()
    
    for r in results:
        # Guardar todas las detecciones para mostrar
        im_array = r.plot()  # imagen con todas las detecciones
        all_detections.append(Image.fromarray(im_array[..., ::-1]))
        
        # Identificar qué clases requeridas fueron detectadas
        for box in r.boxes:
            class_id = int(box.cls[0].item())
            detected_classes.add(class_id)
            if class_id in required_classes:
                required_detected.add(class_id)
    
    return all_detections[0], detected_classes, required_detected

if option == "Subir imagen":
    uploaded_file = st.file_uploader("Elija una imagen...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None and model is not None:
        # Leer la imagen original
        original_image = Image.open(uploaded_file)
        
        with col1:
            st.image(original_image, caption="Imagen original", use_container_width=True)
        
        # Procesar imagen
        detected_image, all_classes, req_detected = process_image(original_image)
        
        with col2:
            st.image(detected_image, caption="Todas las detecciones", use_container_width=True)
        
        # Evaluar cumplimiento de EPP
        missing_classes = set(required_classes) - req_detected
        compliance_status = st.empty()
        
        if not missing_classes:
            compliance_status.success("✅ El trabajador tiene el equipo de protección personal adecuado")
        else:
            missing_names = [CLASS_NAMES[class_id] for class_id in missing_classes]
            compliance_status.error(f"❌ No apto. Faltan: {', '.join(missing_names)}")

else:  # Usar cámara
    picture = st.camera_input("Tome una foto")
    
    if picture and model is not None:
        # Leer la imagen de la cámara
        original_image = Image.open(picture)
        
        with col1:
            st.image(original_image, caption="Imagen de la cámara", use_container_width=True)
        
        # Procesar imagen
        detected_image, all_classes, req_detected = process_image(original_image)
        
        with col2:
            st.image(detected_image, caption="Todas las detecciones", use_container_width=True)
        
        # Evaluar cumplimiento de EPP
        missing_classes = set(required_classes) - req_detected
        compliance_status = st.empty()
        
        if not missing_classes:
            compliance_status.success("✅ El trabajador tiene el equipo de protección personal adecuado")
        else:
            missing_names = [CLASS_NAMES[class_id] for class_id in missing_classes]
            compliance_status.error(f"❌ No apto. Faltan: {', '.join(missing_names)}")

        # Mostrar resumen de detecciones
        st.subheader("Resumen de Detecciones")
        detected_names = [CLASS_NAMES[class_id] for class_id in all_classes if class_id in CLASS_NAMES]
        st.write(f"Elementos detectados: {', '.join(detected_names) if detected_names else 'Ninguno'}")