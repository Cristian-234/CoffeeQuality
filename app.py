from pathlib import Path
import io

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

import streamlit as st
import torch
from torchvision import transforms
import timm


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="CoffeeQuality ViT Compacto",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    BASE_DIR
    / "model"
    / "coffeequality_vit_compact_int8.pth"
)

DEFAULT_MODEL_NAME = "vit_base_patch16_224"
DEFAULT_IMG_SIZE = 224


# ============================================================
# INFORMACIÓN DE LAS CLASES
# ============================================================

CLASS_INFO = {
    "defect": {
        "titulo": "Grano defectuoso",
        "icono": "⚠️",
        "descripcion": (
            "Presenta daños visibles, roturas, coloraciones "
            "anormales, cavidades u otras irregularidades."
        ),
        "recomendacion": (
            "Separar el grano del lote y realizar una inspección "
            "manual complementaria."
        ),
    },
    "longberry": {
        "titulo": "Longberry",
        "icono": "📏",
        "descripcion": (
            "Grano caracterizado por una forma alargada y un "
            "tamaño diferente al convencional."
        ),
        "recomendacion": (
            "Clasificarlo según los criterios morfológicos del "
            "lote y verificar su uniformidad."
        ),
    },
    "peaberry": {
        "titulo": "Peaberry",
        "icono": "🟤",
        "descripcion": (
            "Grano redondeado que se desarrolla individualmente "
            "dentro de la cereza del café."
        ),
        "recomendacion": (
            "Separarlo para una evaluación específica y un posible "
            "tratamiento comercial diferenciado."
        ),
    },
    "premium": {
        "titulo": "Grano premium",
        "icono": "✅",
        "descripcion": (
            "Presenta una apariencia uniforme y no muestra "
            "defectos visuales relevantes."
        ),
        "recomendacion": (
            "Mantenerlo en el lote de mayor calidad y continuar "
            "con controles físicos y sensoriales."
        ),
    },
}


# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 22px;
        background: linear-gradient(
            135deg,
            #2d1b12 0%,
            #5a3826 55%,
            #8a5a36 100%
        );
        color: white;
        box-shadow: 0 14px 34px rgba(45,27,18,.18);
        margin-bottom: 1.3rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.15rem;
    }

    .hero p {
        margin: .55rem 0 0;
        opacity: .92;
    }

    .result-card {
        padding: 1.3rem 1.4rem;
        border-radius: 18px;
        background: white;
        border: 1px solid #eadfd5;
        box-shadow: 0 8px 24px rgba(75,49,32,.08);
    }

    .result-label {
        color: #6e4c37;
        font-size: .88rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .04em;
    }

    .result-title {
        margin-top: .3rem;
        font-size: 1.8rem;
        font-weight: 800;
        color: #2d1b12;
    }

    .confidence {
        margin-top: .4rem;
        font-size: 1.15rem;
        font-weight: 700;
        color: #6a8b47;
    }

    .info-box {
        padding: 1rem 1.1rem;
        border-radius: 14px;
        background: #f8f4ef;
        border-left: 5px solid #8a5a36;
        margin-top: .8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalize_class_to_idx(mapping):
    """Normaliza el mapeo de clases guardado en el checkpoint."""

    if not isinstance(mapping, dict):
        return None

    output = {}

    for class_name, class_index in mapping.items():
        try:
            output[str(class_name)] = int(class_index)
        except (TypeError, ValueError):
            continue

    return output or None


def display_name(class_name):
    """Devuelve el nombre amigable de una clase."""

    return CLASS_INFO.get(
        class_name,
        {}
    ).get(
        "titulo",
        class_name.replace("_", " ").title()
    )


# ============================================================
# CARGA DEL MODELO COMPACTO
# ============================================================

@st.cache_resource(
    show_spinner="Cargando modelo ViT compacto..."
)
def load_compact_model():
    """
    Carga un checkpoint con pesos almacenados en INT8
    personalizado y reconstruye el modelo en FP32 para inferencia.

    No requiere motores de cuantización como x86, fbgemm o qnnpack.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en:\n{MODEL_PATH}\n\n"
            "Coloca 'coffeequality_vit_compact_int8.pth' "
            "dentro de la carpeta 'model'."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "El archivo no contiene un checkpoint válido."
        )

    precision = checkpoint.get("precision", "")

    if precision != "custom_int8_weights":
        raise ValueError(
            "El archivo no corresponde al modelo compacto esperado. "
            f"Precisión detectada: {precision or 'no especificada'}"
        )

    compact_state_dict = checkpoint.get(
        "compact_state_dict"
    )

    if compact_state_dict is None:
        raise ValueError(
            "El checkpoint no contiene 'compact_state_dict'."
        )

    model_name = checkpoint.get(
        "model_name",
        DEFAULT_MODEL_NAME,
    )

    img_size = int(
        checkpoint.get(
            "img_size",
            DEFAULT_IMG_SIZE,
        )
    )

    class_to_idx = normalize_class_to_idx(
        checkpoint.get("class_to_idx")
    )

    if class_to_idx is None:
        class_to_idx = {
            "defect": 0,
            "longberry": 1,
            "peaberry": 2,
            "premium": 3,
        }

    idx_to_class = {
        class_index: class_name
        for class_name, class_index
        in class_to_idx.items()
    }

    # Crear la arquitectura ViT original.
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=len(class_to_idx),
    )

    restored_state_dict = {}

    # Reconstruir cada tensor en float32.
    for parameter_name, parameter_data in compact_state_dict.items():

        if not isinstance(parameter_data, dict):
            raise ValueError(
                f"Formato inválido en el parámetro: {parameter_name}"
            )

        tensor = parameter_data.get("data")

        if tensor is None:
            raise ValueError(
                f"No se encontró 'data' para: {parameter_name}"
            )

        if parameter_data.get("quantized", False):
            scale = float(
                parameter_data.get("scale", 1.0)
            )

            restored_state_dict[parameter_name] = (
                tensor.float() * scale
            )
        else:
            restored_state_dict[parameter_name] = (
                tensor.float()
                if torch.is_floating_point(tensor)
                else tensor
            )

    model.load_state_dict(
        restored_state_dict,
        strict=True,
    )

    model.cpu()
    model.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize(
                (img_size, img_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    info = {
        "model_name": model_name,
        "img_size": img_size,
        "precision": (
            "INT8 almacenado / FP32 en inferencia"
        ),
        "device": "CPU",
        "model_file": MODEL_PATH.name,
    }

    return (
        model,
        preprocess,
        idx_to_class,
        info,
    )


# ============================================================
# PREDICCIÓN
# ============================================================

@torch.inference_mode()
def predict_image(image):
    model, preprocess, idx_to_class, _ = (
        load_compact_model()
    )

    tensor = preprocess(
        image.convert("RGB")
    ).unsqueeze(0)

    outputs = model(tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1,
    )[0].cpu().numpy()

    predicted_index = int(
        np.argmax(probabilities)
    )

    distribution = {
        idx_to_class[index]: float(
            probabilities[index]
        )
        for index in range(
            len(probabilities)
        )
    }

    return (
        idx_to_class[predicted_index],
        float(probabilities[predicted_index]),
        distribution,
    )


# ============================================================
# CABECERA
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>☕ CoffeeQuality ViT</h1>
        <p>
            Clasificación visual de granos de café mediante
            Vision Transformer con almacenamiento compacto INT8.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BARRA LATERAL
# ============================================================

with st.sidebar:
    st.header("Acerca del sistema")

    st.write(
        "La aplicación analiza una imagen de un grano de café "
        "y determina su categoría visual."
    )

    st.divider()

    st.subheader("Clases reconocidas")

    st.markdown(
        "- ⚠️ **Defect**\n"
        "- 📏 **Longberry**\n"
        "- 🟤 **Peaberry**\n"
        "- ✅ **Premium**"
    )

    st.divider()

    st.subheader("Información del modelo")

    try:
        _, _, _, info = load_compact_model()

        st.success(
            "Modelo cargado correctamente"
        )

        st.caption(
            f"Arquitectura: `{info['model_name']}`"
        )

        st.caption(
            f"Formato: `{info['precision']}`"
        )

        st.caption(
            f"Resolución: "
            f"`{info['img_size']} × {info['img_size']}`"
        )

        st.caption(
            f"Ejecución: `{info['device']}`"
        )

        st.caption(
            f"Archivo: `{info['model_file']}`"
        )

    except Exception as error:
        st.error(
            "No se pudo cargar el modelo."
        )

        st.caption(str(error))

    st.divider()

    st.caption(
        "La clasificación visual no reemplaza una evaluación "
        "física, química o sensorial profesional."
    )


# ============================================================
# ENTRADA DE IMAGEN
# ============================================================

upload_tab, camera_tab = st.tabs(
    [
        "📤 Subir imagen",
        "📷 Usar cámara",
    ]
)

uploaded_file = None

with upload_tab:
    uploaded_file = st.file_uploader(
        "Selecciona una fotografía del grano",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        help=(
            "Utiliza una imagen clara, centrada "
            "y con buena iluminación."
        ),
    )

with camera_tab:
    camera_file = st.camera_input(
        "Captura una fotografía del grano"
    )

    if camera_file is not None:
        uploaded_file = camera_file


# ============================================================
# RESULTADO
# ============================================================

if uploaded_file is None:
    st.info(
        "Sube una imagen o utiliza la cámara "
        "para iniciar la clasificación."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Arquitectura",
            "ViT Base",
        )

    with c2:
        st.metric(
            "Formato",
            "INT8 compacto",
        )

    with c3:
        st.metric(
            "Clases",
            "4",
        )

else:
    try:
        image = Image.open(
            io.BytesIO(
                uploaded_file.getvalue()
            )
        ).convert("RGB")

        image_col, result_col = st.columns(
            [1.05, 1],
            gap="large",
        )

        with image_col:
            st.subheader(
                "Imagen analizada"
            )

            st.image(
                image,
                use_container_width=True,
                caption=getattr(
                    uploaded_file,
                    "name",
                    "Captura de cámara",
                ),
            )

        with result_col:
            st.subheader(
                "Resultado"
            )

            with st.spinner(
                "Analizando el grano de café..."
            ):
                (
                    pred_class,
                    confidence,
                    distribution,
                ) = predict_image(image)

            data = CLASS_INFO.get(
                pred_class,
                {
                    "titulo": display_name(
                        pred_class
                    ),
                    "icono": "☕",
                    "descripcion": (
                        "Categoría detectada por el modelo."
                    ),
                    "recomendacion": (
                        "Realizar una revisión visual "
                        "complementaria."
                    ),
                },
            )

            st.markdown(
                f"""
            <div class="result-card">
                <div class="result-label">Clase predicha</div>
                <div class="result-title">
                    {data['icono']} {data['titulo']}
                </div>
                <div class="confidence">
                    Confianza: {confidence * 100:.2f} %
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            if confidence < 0.60:
                st.warning(
                    "La confianza es baja. Utiliza una imagen "
                    "más clara, centrada y con mejor iluminación."
                )

            elif confidence < 0.80:
                st.info(
                    "La predicción tiene confianza moderada. "
                    "Se recomienda revisión visual adicional."
                )

            else:
                st.success(
                    "La predicción presenta un nivel "
                    "de confianza alto."
                )

            st.markdown(
                f"""
            <div class="info-box">
                <strong>Descripción:</strong><br>
                {data['descripcion']}
            </div>

            <div class="info-box">
                <strong>Recomendación:</strong><br>
                {data['recomendacion']}
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.divider()

        st.subheader(
            "Probabilidades por categoría"
        )

        probability_df = pd.DataFrame(
            {
                "Clase": [
                    display_name(class_name)
                    for class_name
                    in distribution.keys()
                ],
                "Probabilidad": [
                    probability * 100
                    for probability
                    in distribution.values()
                ],
            }
        ).sort_values(
            "Probabilidad",
            ascending=False,
        )

        st.bar_chart(
            probability_df.set_index(
                "Clase"
            ),
            horizontal=True,
            use_container_width=True,
        )

        probability_table = (
            probability_df.copy()
        )

        probability_table[
            "Probabilidad"
        ] = probability_table[
            "Probabilidad"
        ].map(
            lambda value: f"{value:.2f} %"
        )

        st.dataframe(
            probability_table,
            hide_index=True,
            use_container_width=True,
        )

    except UnidentifiedImageError:
        st.error(
            "El archivo seleccionado no es "
            "una imagen válida."
        )

    except FileNotFoundError as error:
        st.error(str(error))

        st.code(
            """
DL_U2/
├── app2.py
└── model/
    └── coffeequality_vit_compact_int8.pth
            """
        )

    except RuntimeError as error:
        st.error(
            "No se pudo cargar o ejecutar el modelo compacto. "
            "Verifica que haya sido generado con el fragmento "
            "de cuantización personalizada."
        )

        st.exception(error)

    except Exception as error:
        st.error(
            "Ocurrió un error durante la clasificación."
        )

        st.exception(error)