## Usage

> **Prerequisite:**  
> Make sure [Docker Desktop](https://docs.docker.com/desktop/) is installed on your system.

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

> All components (MLOps pipeline, frontend, backend, Grafana, PostgreSQL, Redis, and MLflow) will be installed and started automatically.  
> Please ensure you have **at least 7–8 GB of free disk space**.  
> Installation time may vary depending on your system performance.  
> The services are **fully operational once you see the message** `PIPELINE READY` appear **five times** in your terminal.
> Once all services are running, you can use the web interface to register, log in, and interact with the heart disease prediction model.

4. Access the services via the following URLs:  
- **Streamlit UI:** [http://localhost:8501/](http://localhost:8501/)  
- **Grafana Dashboard:** [http://localhost:3000/](http://localhost:3000/)  
  - Username: `admin`  
  - Password: `meinpasswort`
- **FastAPI Documentation (Swagger UI):** [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)  
- **MLflow UI:** [http://localhost:5000/](http://localhost:5000/)
