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
DISPLAY_WIDTH = 400  # Ancho máximo para visualización (sin afectar calidad)

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

# Función para procesar la imagen
def process_image(image):
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    results = model.predict(image_cv, imgsz=640)
    
    im_array = results[0].plot()
    detected_image = Image.fromarray(im_array[..., ::-1])
    
    detected_classes = {int(box.cls[0].item()) for r in results for box in r.boxes}
    required_detected = detected_classes & set(required_classes)
    
    return detected_image, detected_classes, required_detected

# Columnas principales
left_col, right_col = st.columns([1, 2])

# Columna izquierda: Configuración
with left_col:
    st.header("Configuración")
    
    option = st.radio("Fuente de imagen:", 
                     ("Subir imagen", "Usar cámara"),
                     horizontal=True)
    
    st.subheader("Elementos Requeridos")
    required_classes = []
    for class_id, class_name in CLASS_NAMES.items():
        if st.checkbox(class_name, value=True, key=f"class_{class_id}"):
            required_classes.append(class_id)

    if option == "Subir imagen":
        uploaded_file = st.file_uploader("Seleccione imagen:", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = None
        picture = st.camera_input("Tome una foto", key="camera_input")

# Columna derecha: Resultados
with right_col:
    st.header("Resultados")
    
    if (uploaded_file or (option == "Usar cámara" and picture)) and model:
        original_image = Image.open(uploaded_file if option == "Subir imagen" else picture)
        
        detected_image, all_classes, req_detected = process_image(original_image)
        
        # Evaluación de cumplimiento
        missing_classes = set(required_classes) - req_detected
        if not missing_classes:
            st.success("✅ **Cumple con todo el equipo requerido**")
        else:
            missing_names = [CLASS_NAMES[c] for c in missing_classes]
            st.error(f"❌ **Faltan elementos:** {', '.join(missing_names)}")
        
        # Mostrar imágenes con tamaño controlado pero manteniendo calidad
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, 
                   caption="Imagen Original",
                   width=DISPLAY_WIDTH)
        with col2:
            st.image(detected_image,
                   caption="Detecciones",
                   width=DISPLAY_WIDTH)
        
        # Resumen de detecciones
        with st.expander("Detalles de detección"):
            st.write(f"**Total detectado:** {len(all_classes)} elementos")
            st.write(f"**Requeridos detectados:** {len(req_detected)} de {len(required_classes)}")
            st.write("**Items detectados:**", ", ".join([CLASS_NAMES[c] for c in all_classes]))

# Estilos CSS para mejorar la presentación
st.markdown("""
<style>
    [data-testid=stImage] {
        text-align: center;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)