# CoffeeQuality ViT — Streamlit

Aplicación web para probar el modelo `vit_base_patch16_224` entrenado para clasificar:

- `defect`
- `longberry`
- `peaberry`
- `premium`

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

## Ejecutar

```bash
streamlit run app.py
```

La aplicación se abrirá normalmente en:

```text
http://localhost:8501
```
