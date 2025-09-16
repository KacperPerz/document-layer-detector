from fastapi import FastAPI
from contextlib import asynccontextmanager
from .core.model import get_model
from .api.endpoints import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    get_model()
    yield

app = FastAPI(title="Document Layout Detector API", lifespan=lifespan)

app.include_router(api_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Document Layout Detector API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}
