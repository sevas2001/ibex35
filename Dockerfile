FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código y modelos
COPY . .

# Puerto
EXPOSE 7860

# Arrancar servidor (puerto 7860 requerido por Hugging Face Spaces)
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]
