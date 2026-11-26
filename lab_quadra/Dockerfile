FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends poppler-utils && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY lab_quadra/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY lab_quadra/app.py /app/lab_quadra/app.py
COPY lab_quadra/report_pdf.py /app/lab_quadra/report_pdf.py
CMD ["sh","-c","streamlit run lab_quadra/app.py --server.address 0.0.0.0 --server.port ${PORT:-10000} --server.headless true --browser.gatherUsageStats false"]
