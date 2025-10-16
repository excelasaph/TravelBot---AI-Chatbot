# Minimal Dockerfile for running TravelBot Streamlit app in a Hugging Face Space (Docker template)
# Uses official slim Python image to keep size small. The Space build will run `docker build`.

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Copy and install dependencies first for better caching. Prefer app/requirements.txt if present.
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY . /app

# Expose the port expected by Spaces
EXPOSE 8080
ENV PYTHONUNBUFFERED=1

# Use the environment PORT that Spaces provides; fallback to 8080
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port ${PORT:-8080} --server.address 0.0.0.0"]
