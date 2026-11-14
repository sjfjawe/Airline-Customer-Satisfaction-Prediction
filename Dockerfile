FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 9696
CMD ["gunicorn", "--bind=0.0.0.0:9696", "serve:app"]
