# Trabajo-1-Analisis-
Trabajo 1 Analisis de Algoritmos
Matías Villarroel Benvente
Pablo Tapia Curimil

# 1. Permitir ejecución de scripts para el usuario actual
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 2. Crear el entorno virtual (solo la primera vez)
python -m venv .venv

# 3. Activar el entorno virtual
.\.venv\Scripts\Activate.ps1

# 4. Instalar librerías necesarias dentro del entorno virtual
pip install numpy scipy matplotlib

# 5. Ejecutar el script principal
python main.py

# 6. (Opcional) Desactivar el entorno virtual cuando termines
deactivate
