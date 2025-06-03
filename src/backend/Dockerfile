FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r src/backend/backend_requirements.txt && pip install . 

CMD ["fastapi", "run", "src/backend/app.py"]