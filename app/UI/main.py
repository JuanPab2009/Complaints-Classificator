import streamlit as st
import requests

# Configurar la URL de la API
API_URL = "http://model:8000/predict"  # Usamos 'model' porque así se llama el servicio en docker-compose

st.set_page_config(
    page_title="Clasificación de Quejas",
    page_icon="💬",
    layout="centered",
)

st.markdown(
    """
    <style>
        .title {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            color: #1a2b5c;
            margin-bottom: 1rem;
        }
        .stApp {
            background-image: linear-gradient(to bottom right, #fdfbfb, #e0f7fa);
            background-attachment: fixed;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='title'>Clasificación de Quejas de Clientes</div>",
    unsafe_allow_html=True,
)

# Descripción
st.write("""
Ingrese la descripción de la queja del cliente en el campo de texto a continuación, y la aplicación clasificará automáticamente la queja en la categoría correspondiente.
""")

# Campo de entrada de texto
complaint_text = st.text_area("Descripción de la Queja", height=200)

# Botón para realizar la predicción
if st.button("Clasificar Queja"):
    if complaint_text.strip() == "":
        st.warning("Por favor, ingrese la descripción de la queja.")
    else:
        # Datos a enviar a la API
        data = {"complaint_what_happened": complaint_text}

        # Realizar la solicitud POST a la API
        try:
            response = requests.post(API_URL, json=data)
            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction", "No se pudo obtener la predicción.")
                st.success(f"La queja ha sido clasificada como: **{prediction}**")
            else:
                st.error(f"Error en la API: {response.status_code} - {response.reason}")
        except Exception as e:
            st.error(f"Ocurrió un error al comunicarse con la API: {e}")
