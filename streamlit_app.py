import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
import urllib.request
from ultralytics import YOLO

# Configuración de la página con fondo GIF
st.set_page_config(page_title="Detección de EPP", layout="wide")

# Añadir CSS para el fondo GIF
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://media.licdn.com/dms/image/v2/D5610AQEQGynBerz0qw/image-shrink_800/image-shrink_800/0/1713283207488?e=2147483647&v=beta&t=WQvr-qzbeq4uxBjoEp5WfJcv7FdLcAtodhDH8PZJObg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    /* Fondo semitransparente para el contenido */
    .st-emotion-cache-1y4p8pa, 
    .st-emotion-cache-16txtl3,
    .st-emotion-cache-1pxazr7,
    .st-emotion-cache-1q7spjk {
        background-color: rgba(255, 255, 255, 0.85) !important;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Mejorar legibilidad del texto */
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #333333 !important;
    }
    
    /* Ajustar imágenes de detección */
    [data-testid=stImage] {
        background-color: white;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título de la aplicación con fondo semitransparente
st.markdown(
    """
    <div style="background-color: rgba(255, 255, 255, 0.8); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: #333333; text-align: center;">🚨 Sistema de Detección de Equipo de Protección Personal</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Configuración del modelo
MODEL_PATH = "model/best.pt"
CLASS_NAMES = {0: "⛑️ Casco", 1: "👞 Zapatos", 2: "👤 Persona", 3: "🦺 Chaleco"}
DISPLAY_WIDTH = 400  # Ancho de visualización sin perder calidad

@st.cache_resource
def load_model():
    try:
        os.makedirs("model", exist_ok=True)
        
        if not os.path.exists(MODEL_PATH):
            with st.spinner('⏳ Descargando el modelo... Esto puede tomar unos minutos'):
                urllib.request.urlretrieve(MODEL_PATH)
                st.success("✔️ Modelo descargado exitosamente!")
        
        model = YOLO(MODEL_PATH)
        return model
    
    except Exception as e:
        st.error(f"❌ No se pudo cargar el modelo: {str(e)}")
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
    with st.container():
        st.header("⚙️ Configuración")
        
        option = st.radio("📷 Seleccione fuente de imagen:", 
                         ("📤 Subir imagen", "📸 Usar cámara"),
                         horizontal=True)
        
        st.subheader("🛡️ Elementos Requeridos")
        required_classes = []
        for class_id, class_name in CLASS_NAMES.items():
            # No mostramos la opción de persona ya que siempre se detectará
            if class_id != 2:  # 2 es el ID de persona
                if st.checkbox(class_name, value=True, key=f"class_{class_id}"):
                    required_classes.append(class_id)

        if option == "📤 Subir imagen":
            uploaded_file = st.file_uploader("🖼️ Seleccione imagen:", type=["jpg", "jpeg", "png"])
        else:
            uploaded_file = None
            picture = st.camera_input("📸 Tome una foto", key="camera_input")

# Columna derecha: Resultados
with right_col:
    with st.container():
        st.header("🔍 Resultados del Análisis")
        
        if (uploaded_file or (option == "📸 Usar cámara" and picture)) and model:
            original_image = Image.open(uploaded_file if option == "📤 Subir imagen" else picture)
            
            detected_image, all_classes, req_detected = process_image(original_image)
            
            # Evaluación de cumplimiento
            missing_classes = set(required_classes) - req_detected
            if not missing_classes:
                st.success("✅ **¡Cumple con todo el equipo requerido!**")
            else:
                missing_names = [CLASS_NAMES[c].split()[-1] for c in missing_classes]  # Remueve emoji para mostrar solo nombre
                st.error(f"❌ **¡No apto! Faltan:** {', '.join(missing_names)}")
            
            # Mostrar imágenes con tamaño controlado manteniendo calidad
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_image, 
                       caption="🖼️ Imagen Original",
                       width=DISPLAY_WIDTH)
            with col2:
                st.image(detected_image,
                       caption="🔎 Detecciones",
                       width=DISPLAY_WIDTH)
            
            # Resumen de detecciones
            with st.expander("📊 Detalles completos de detección"):
                st.write(f"**📌 Total detectado:** {len(all_classes)} elementos")
                st.write(f"**✅ Requeridos detectados:** {len(req_detected)} de {len(required_classes)}")
                detected_items = [CLASS_NAMES[c] for c in all_classes if c in CLASS_NAMES]
                st.write("**🔍 Items detectados:**", ", ".join(detected_items) if detected_items else "Ninguno")
        else:
            st.info("ℹ️ Configure los parámetros y cargue una imagen para realizar el análisis")