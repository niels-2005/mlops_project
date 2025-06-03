FROM python:3.10

WORKDIR /app

COPY . . 

RUN pip install --no-cache-dir -r src/frontend/frontend_requirements.txt

COPY . .

CMD ["streamlit", "run", "src/frontend/main.py"]