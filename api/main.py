from fastapi import FastAPI

app = FastAPI(
    title="FinWise Credit Risk API",
    description="AI-powered credit risk scoring API",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "FinWise Credit Risk API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
