from fastapi import FastAPI

app = FastAPI(title="Document Layout Detector API")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Document Layout Detector API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}
