FROM python:3.11-slim

# System deps for OCR + PDF rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-fra \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./app.py
EXPOSE 10000

# Streamlit config: listen on all interfaces
CMD ["bash", "-lc", "python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true --server.fileWatcherType=none --browser.gatherUsageStats=false"]

