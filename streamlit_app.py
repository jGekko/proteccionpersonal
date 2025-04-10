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
CLASS_NAMES = {0: "Casco", 1: "Zapatos", 2: "Persona", 3: "Chaleco"}
MAX_IMAGE_SIZE = 300  # Tamaño máximo para las imágenes

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

# Función para redimensionar imágenes manteniendo el aspect ratio
def resize_image(image, max_size):
    width, height = image.size
    ratio = min(max_size/width, max_size/height)
    new_size = (int(width*ratio), int(height*ratio))
    return image.resize(new_size, Image.LANCZOS)

# Función para procesar la imagen y obtener resultados
def process_image(image):
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    results = model.predict(image_cv, imgsz=640)
    
    # Imagen con todas las detecciones
    im_array = results[0].plot()
    detected_image = Image.fromarray(im_array[..., ::-1])
    
    # Clases detectadas
    detected_classes = {int(box.cls[0].item()) for r in results for box in r.boxes}
    required_detected = detected_classes & set(required_classes)
    
    return detected_image, detected_classes, required_detected

# Crear columnas principales
left_col, right_col = st.columns([1, 2])

# Columna izquierda: Configuración
with left_col:
    st.header("Configuración")
    
    # Selección de fuente de imagen
    option = st.radio("Seleccione fuente de imagen:", 
                     ("Subir imagen", "Usar cámara"),
                     index=0)
    
    # Selector de EPP requerido
    st.subheader("Elementos de Protección Requeridos")
    required_classes = []
    for class_id, class_name in CLASS_NAMES.items():
        if st.checkbox(class_name, value=True, key=f"class_{class_id}"):
            required_classes.append(class_id)
    
    # Espacio para cargar imagen (si se selecciona esa opción)
    if option == "Subir imagen":
        uploaded_file = st.file_uploader("Seleccione una imagen:", 
                                       type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = None
        picture = st.camera_input("Tome una foto con la cámara")

# Columna derecha: Resultados
with right_col:
    st.header("Resultados del Análisis")
    
    if (uploaded_file is not None or (option == "Usar cámara" and picture is not None)) and model is not None:
        # Obtener la imagen según la fuente seleccionada
        if option == "Subir imagen":
            original_image = Image.open(uploaded_file)
        else:
            original_image = Image.open(picture)
        
        # Procesar la imagen
        detected_image, all_classes, req_detected = process_image(original_image)
        
        # Evaluar cumplimiento
        missing_classes = set(required_classes) - req_detected
        if not missing_classes:
            st.success("✅ **Resultado:** El trabajador tiene todo el equipo de protección requerido")
        else:
            missing_names = [CLASS_NAMES[c] for c in missing_classes]
            st.error(f"❌ **Resultado:** No apto. Faltan: {', '.join(missing_names)}")
        
        # Mostrar imágenes en un grid
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(resize_image(original_image, MAX_IMAGE_SIZE), 
                   caption="Imagen Original",
                   use_container_width=True)
        with img_col2:
            st.image(resize_image(detected_image, MAX_IMAGE_SIZE),
                   caption="Análisis de Detección",
                   use_container_width=True)
        
        # Resumen de detecciones
        st.subheader("Resumen de Detecciones")
        detected_names = [CLASS_NAMES[c] for c in all_classes if c in CLASS_NAMES]
        st.write(f"**Elementos detectados:** {', '.join(detected_names) if detected_names else 'Ninguno'}")
        st.write(f"**Elementos requeridos detectados:** {', '.join([CLASS_NAMES[c] for c in req_detected]) if req_detected else 'Ninguno'}")
    else:
        st.info("ℹ️ Configure los parámetros y cargue una imagen para realizar el análisis")

# Estilos CSS personalizados
st.markdown("""
<style>
    .stImage img {
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stRadio > div {
        display: flex;
        gap: 10px;
    }
    .stRadio [role=radiogroup] {
        gap: 5px;
    }
</style>
""", unsafe_allow_html=True)