from pydantic import BaseModel
import mlflow
from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
import logging
import nltk
import re
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de MLFLOW
MLFLOW_TRACKING_URI = 'https://dagshub.com/zapatacc/final-exam-pcd2024-autumn.mlflow'
mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)

#
app = FastAPI()


class InputData(BaseModel):
    complaint_what_happened: str


# Función para obtener el run_id del Champion
def get_run_id_from_champion(model_name: str, alias: str) -> str:
    client = mlflow.tracking.MlflowClient()
    model_version = client.get_model_version_by_alias(name=model_name, alias=alias)
    return model_version.run_id


# Funciones de preprocesamiento
def expand_contractions(text):
    return contractions.fix(text)


def clean_text(text):
    text = expand_contractions(text)
    text = text.lower()
    text = re.sub(r'xx+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_text(text):
    text = clean_text(text)
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(words)
    return text


# Cargar el modelo y los artefactos necesarios
def load_model_and_artifacts():
    try:
        model_name = "blanco-models-prefect"
        alias = "champion"
        run_id = get_run_id_from_champion(model_name=model_name, alias=alias)
        logged_model = f"runs:/{run_id}/model"
        # Cargar el modelo desde MLflow
        loaded_model = mlflow.pyfunc.load_model(model_uri=logged_model)

        # Descargar y cargar el LabelEncoder
        label_encoder_path = mlflow.artifacts.download_artifacts(run_id=run_id,
                                                                 artifact_path="LabelEncoder/label_encoder.pkl")
        with open(label_encoder_path, "rb") as f:
            label_encoder = pickle.load(f)

        # Descargar y cargar el TF-IDF Vectorizer
        tfidf_vectorizer_path = mlflow.artifacts.download_artifacts(run_id=run_id,
                                                                    artifact_path="Vectorizer/tfidf_vectorizer.pkl")
        with open(tfidf_vectorizer_path, "rb") as f:
            tfidf_vectorizer = pickle.load(f)

        logger.info("Modelo, LabelEncoder y Vectorizer cargados correctamente.")
        return loaded_model, label_encoder, tfidf_vectorizer

    except Exception as e:
        logger.error(f"Error al cargar el modelo y los artefactos: {e}")
        raise e


# Cuando inicie la aplicación cargamos el modelo, el LabelEncoder y el Vectorizer
model, label_encoder, tfidf_vectorizer = load_model_and_artifacts()


# Definimos el endpoint de predicción
@app.post("/predict")
def predict_endpoint(input_data: InputData):
    try:
        # Convertimos los datos de entrada en un diccionario y luego en un dataframe
        data = input_data.dict()
        df = pd.DataFrame([data])

        # Extraemos el texto de la queja
        text = df["complaint_what_happened"].iloc[0]

        # Preprocesamos el texto
        processed_text = preprocess_text(text)

        # Vectorizamos el texto
        text_tfidf = tfidf_vectorizer.transform([processed_text])

        # Hacemos predicción con el modelo cargado
        prediction_encoded = model.predict(text_tfidf)

        # Decodificamos la predicción usando el LabelEncoder para obtener la etiqueta original
        prediction_decoded = label_encoder.inverse_transform(prediction_encoded.astype(int))

        # Retornamos la predicción como JSON
        return {
            "prediction": prediction_decoded[0]
        }

    except Exception as e:
        logger.error(f"Error durante la predicción: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor al procesar la predicción.")
