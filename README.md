## Verwendung

> **Voraussetzung:**  
> Bitte Sicherstellen, dass [Docker Desktop](https://docs.docker.com/desktop/) auf Ihrem System installiert ist und läuft.

1. Repository klonen:  
```bash
git clone git@github.com:niels-2005/mlops_project.git
```

2. In das Projektverzeichnis wechseln:  
```bash
cd mlops_project/
```
3. Services mit Docker Compose erstellen und starten:  
```bash
docker compose up --build   
```

> Alle Komponenten (MLOps-Pipeline, Frontend, Backend, Grafana, PostgreSQL, Redis und MLflow) werden automatisch installiert und gestartet.  
> Bitte sicherstellen , dass Sie **mindestens 7–8 GB freien Festplattenspeicher** haben.  
> Die Services sind **voll funktionsfähig, sobald die Nachricht** `PIPELINE READY` **fünfmal** im Terminal erscheint.
> Sobald alle Services laufen, können Sie die Web-Oberfläche verwenden, um sich zu registrieren, anzumelden und mit dem Vorhersagemodell für Herzkrankheiten zu interagieren.
> Die Schema Dateien um den MLOps-Workflow zu konfigurieren sind unter src/mlops/schemas zu finden.

4. Zugriff auf die Services über die folgenden URLs:  
- **Streamlit UI:** [http://localhost:8501/](http://localhost:8501/)  
- **Grafana Dashboard:** [http://localhost:3000/](http://localhost:3000/)  
  - Benutzername: `admin`  
  - Passwort: `meinpasswort`
- **FastAPI Dokumentation (Swagger UI):** [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)  
- **MLflow UI:** [http://localhost:5000/](http://localhost:5000/)

