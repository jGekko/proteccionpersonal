import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
import urllib.request
from ultralytics import YOLO

# Configuración de la página
st.set_page_config(page_title="Detección de EPP", layout="centered")

# Título de la aplicación
st.title("Sistema de Detección de Equipo de Protección Personal")

# Configuración del modelo
MODEL_PATH = "model/best.pt"
CLASS_NAMES = {0: "casco", 1: "zapatos", 2: "persona", 3: "chaleco"}
MAX_IMAGE_SIZE = 600  # Tamaño máximo en píxeles para mostrar las imágenes

@st.cache_resource
def load_model():
    try:
        os.makedirs("model", exist_ok=True)
        
        if not os.path.exists(MODEL_PATH):
            with st.spinner('Descargando el modelo... Esto puede tomar unos minutos'):
                urllib.request.urlretrieve(MODEL_PATH)
                st.success("Modelo descargado exitosamente!")
        
        model = YOLO(MODEL_PATH)
        return model
    
    except Exception as e:
        st.error(f"No se pudo cargar el modelo: {str(e)}")
        return None

model = load_model()

# Sidebar para selección de EPP requerido
with st.sidebar:
    st.header("Configuración de Detección")
    st.markdown("Seleccione los elementos de protección requeridos:")
    
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

def resize_image(image, max_size):
    """Redimensiona la imagen manteniendo el aspect ratio"""
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    return image.resize((new_width, new_height))

def process_image(image):
    """Procesa la imagen y devuelve resultados"""
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    results = model.predict(image_cv, imgsz=640)
    
    all_detections = []
    detected_classes = set()
    required_detected = set()
    
    for r in results:
        im_array = r.plot()
        all_detections.append(Image.fromarray(im_array[..., ::-1]))
        
        for box in r.boxes:
            class_id = int(box.cls[0].item())
            detected_classes.add(class_id)
            if class_id in required_classes:
                required_detected.add(class_id)
    
    return all_detections[0], detected_classes, required_detected

# Contenedor principal
if option == "Subir imagen":
    uploaded_file = st.file_uploader("Elija una imagen...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None and model is not None:
        original_image = Image.open(uploaded_file)
        
        # Procesar imagen
        detected_image, all_classes, req_detected = process_image(original_image)
        
        # Evaluar cumplimiento de EPP (ahora arriba de las imágenes)
        missing_classes = set(required_classes) - req_detected
        compliance_status = st.container()
        
        if not missing_classes:
            compliance_status.success("✅ El trabajador tiene el equipo de protección personal adecuado")
        else:
            missing_names = [CLASS_NAMES[class_id] for class_id in missing_classes]
            compliance_status.error(f"❌ No apto. Faltan: {', '.join(missing_names)}")
        
        # Mostrar imágenes redimensionadas
        col1, col2 = st.columns(2)
        with col1:
            st.image(resize_image(original_image, MAX_IMAGE_SIZE), 
                    caption="Imagen original", 
                    use_container_width=True)
        with col2:
            st.image(resize_image(detected_image, MAX_IMAGE_SIZE), 
                    caption="Todas las detecciones", 
                    use_container_width=True)

else:  # Usar cámara
    picture = st.camera_input("Tome una foto")
    
    if picture and model is not None:
        original_image = Image.open(picture)
        
        # Procesar imagen
        detected_image, all_classes, req_detected = process_image(original_image)
        
        # Evaluar cumplimiento de EPP (arriba de las imágenes)
        missing_classes = set(required_classes) - req_detected
        compliance_status = st.container()
        
        if not missing_classes:
            compliance_status.success("✅ El trabajador tiene el equipo de protección personal adecuado")
        else:
            missing_names = [CLASS_NAMES[class_id] for class_id in missing_classes]
            compliance_status.error(f"❌ No apto. Faltan: {', '.join(missing_names)}")
        
        # Mostrar imágenes redimensionadas
        col1, col2 = st.columns(2)
        with col1:
            st.image(resize_image(original_image, MAX_IMAGE_SIZE), 
                    caption="Imagen de la cámara", 
                    use_container_width=True)
        with col2:
            st.image(resize_image(detected_image, MAX_IMAGE_SIZE), 
                    caption="Todas las detecciones", 
                    use_container_width=True)

        # Resumen de detecciones
        st.subheader("Resumen de Detecciones")
        detected_names = [CLASS_NAMES[class_id] for class_id in all_classes if class_id in CLASS_NAMES]
        st.write(f"Elementos detectados: {', '.join(detected_names) if detected_names else 'Ninguno'}")