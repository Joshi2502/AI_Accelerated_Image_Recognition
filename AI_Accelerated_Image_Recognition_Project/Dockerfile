
# Dockerfile for AI-Accelerated Image Recognition System
FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN pip install flask torch torchvision pillow

EXPOSE 5000

CMD ["python", "app.py"]
