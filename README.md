Aquí tienes la guía paso a paso lista para compartir con tu equipo o pegar directamente en el archivo `README.md` de tu repositorio de GitHub.

Está diseñada para prevenir errores de versión (como el de Python 3.14 con TensorFlow) y asegurar que todos puedan levantar la app en Streamlit y el Notebook en pocos minutos.

---

# 🏫 Guía de Instalación y Ejecución Local

Este proyecto permite clasificar y evaluar el nivel de riesgo de deserción escolar en Unidades Educativas de Ecuador utilizando datos abiertos del MINEDUC, un modelo propio de **Perceptrón Multicapa (MLP en Keras)**, 5 modelos comparativos de línea base e interpretabilidad mediante **SHAP**.

---

## 📋 1. Requisitos Previos

* **Python 3.11** (Recomendado para asegurar compatibilidad total con TensorFlow 2.x).
* **Git** instalado en la computadora.

---

## 🚀 2. Paso a Paso para la Instalación

### Paso 1: Clonar el Repositorio

Abre tu terminal (PowerShell o CMD) y ejecuta:

```bash
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO

```

---

### Paso 2: Crear y Activar el Entorno Virtual (`.venv`)

* **En Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

```


*(Si da error de permisos en PowerShell, ejecuta primero: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y vuelve a activar).*
* **En macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate

```



*(Verás el prefijo `(.venv)` al inicio de la línea de tu terminal).*

---

### Paso 3: Instalar las Dependencias

Asegúrate de tener la última versión de `pip` e instala las librerías necesarias:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt ipykernel

```

---

### Paso 4: Verificar los Archivos de Datos

Asegúrate de que los dos archivos de Excel descargados de la página del MINEDUC estén ubicados en la ruta `data/raw/`:

* `data/raw/1Registro-Administrativo-Historico_2009-202X-Inicio.xlsx`
* `data/raw/2Registro-Administrativo-Historico_2009-2024-Fin.xlsx`

---

## 💻 3. Opciones de Ejecución y Prueba

### Opción A: Levantar la Aplicación Web (Streamlit)

Para interactuar con el **Semáforo de Riesgo por AMIE**, la **Tabla Comparativa** y la **Explicabilidad SHAP**:

```bash
streamlit run app.py

```

*(Se abrirá automáticamente una pestaña en tu navegador en `http://localhost:8501`).*

---

### Opción B: Ejecutar el Notebook de Experimentos

1. Abre **VS Code** en la carpeta del proyecto.
2. Abre el archivo `notebooks/Entrenatorio_y_Evaluacion.ipynb`.
3. En la esquina superior derecha de VS Code, selecciona el Kernel `(.venv) Python 3.11`.
4. Haz clic en **"Run All"** para entrenar los 6 modelos y visualizar las curvas de pérdida, matrices de confusión y gráficos SHAP.

*También puedes ejecutar todo el notebook directamente desde la terminal con:*

```bash
jupyter nbconvert --to notebook --execute notebooks/Entrenatorio_y_Evaluacion.ipynb --inplace

```

---

### Opción C: Scripts de Prueba Rápida

Puedes validar cada fase del sistema individualmente ejecutando:

* **Prueba de ETL y Preprocesamiento:**
```bash
python test_fase1.py

```


* **Prueba de Entrenamiento y Comparativa (6 Modelos):**
```bash
python test_fase2.py

```


* **Prueba de Explicabilidad (SHAP):**
```bash
python test_shap.py

```



---

## 🛠️ Solución a Problemas Frecuentes

* **Error `No module named 'tensorflow'` o incompatibilidad de versión:**
Asegúrate de estar usando el entorno `.venv` creado con **Python 3.11** y de haberlo activado correctamente antes de correr `pip install`.
* **Procesos colgados en VS Code al ejecutar celdas:**
Ve al menú superior de Jupyter en VS Code y selecciona **"Restart Kernel"**, o cierra los procesos de Python en la terminal con `taskkill /F /IM python.exe /T` (en Windows) y vuelve a intentar.
