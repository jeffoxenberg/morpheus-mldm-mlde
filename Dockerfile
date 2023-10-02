FROM python:3.10-slim
COPY requirements.txt .
COPY launcher.py .
RUN apt update && apt install -y git-all
RUN pip install -r requirements.txt