FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Default: start the server
# Override with: docker run ... python inference.py medium
CMD ["python", "server.py"]
