FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# Starts inference.py in HTTP server mode for OpenEnv checker
# Override for CLI: docker run ... python inference.py medium
CMD ["python", "inference.py"]
