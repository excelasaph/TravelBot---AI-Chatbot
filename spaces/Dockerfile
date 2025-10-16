FROM python:3.10-slim
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app
# Copy only the requirements first (improves cache hits) and install
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
# Copy the rest of the application
COPY . /app
EXPOSE 8080
ENV PYTHONUNBUFFERED=1
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port ${PORT:-8080} --server.address 0.0.0.0"]
