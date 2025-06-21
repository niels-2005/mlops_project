## Usage

1. Clone the repository:  
```bash
git clone git@github.com:niels-2005/mlops_project.git
```

2. Navigate into the project directory:  
```bash
cd mlops_project/
```

3. Build and start the services using Docker Compose:  
```bash
docker compose up --build
```

4. Access the services via the following URLs:  
- **Streamlit UI:** [http://localhost:8501/](http://localhost:8501/)  
- **Grafana Dashboard:** [http://localhost:3000/](http://localhost:3000/)  
- **FastAPI Documentation:** [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)  
- **MLflow UI:** [http://localhost:5000/](http://localhost:5000/)