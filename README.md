# CoffeeQuality ViT — Streamlit

Aplicación web para probar el modelo `vit_base_patch16_224` entrenado para clasificar:

- `defect`
- `longberry`
- `peaberry`
- `premium`

## 1. Estructura del proyecto

```text
CoffeeQuality_Streamlit/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
└── model/
    ├── coffeequality_vit_checkpoint.pth
    └── label_mapping.json            # opcional
```

## 2. Colocar el modelo

Copia dentro de `model/` el archivo generado por Colab:

```text
coffeequality_vit_checkpoint.pth
```

También se admite `coffeequality_vit_best.pth`.

No es necesario incluir a la vez `state_dict.pth` y `checkpoint.pth`.

## 3. Crear entorno virtual

### Windows

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Ejecutar

```bash
streamlit run app.py
```

La aplicación se abrirá normalmente en:

```text
http://localhost:8501
```

## 5. Consideraciones

- La primera carga puede tardar debido al tamaño del ViT.
- En CPU, la predicción puede demorar algunos segundos.
- La aplicación usa la misma normalización del entrenamiento:
  - media: `[0.485, 0.456, 0.406]`
  - desviación estándar: `[0.229, 0.224, 0.225]`
- El checkpoint debe contener `model_state_dict`, `class_to_idx`, `model_name` e `img_size`.
- Para desplegar en Streamlit Community Cloud, un archivo de 327 MB puede superar las restricciones prácticas de GitHub. En ese caso conviene almacenar el modelo en Hugging Face Hub, Google Drive o usar otra plataforma de despliegue.
