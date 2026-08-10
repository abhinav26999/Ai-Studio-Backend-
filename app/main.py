import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.db.firebase import initialize_firebase
from app.routes.wallet import router as wallet_router
from app.routes.jobs import router as jobs_router
from app.routes.webhooks import router as webhooks_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ai_studio")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing AI Studio Backend Services...")
    try:
        initialize_firebase()
        logger.info(f"Firebase Admin SDK initialized successfully for project: {settings.FIREBASE_PROJECT_ID}")
    except Exception as err:
        logger.error(f"Failed to initialize Firebase: {str(err)}")
    yield
    logger.info("Shutting down AI Studio Backend Services...")

app = FastAPI(
    title="AI Studio - Phase 1 Fast Tier Backend API",
    description=(
        "Backend orchestration API for credit management, prompt routing via Hugging Face Qwen 2.5 7B, "
        "Cloud Tasks queueing, and Cloud Run GPU worker webhooks."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for Flutter Client access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(wallet_router)
app.include_router(jobs_router)
app.include_router(webhooks_router)

@app.get("/health", tags=["system"])
async def health_check():
    return {
        "status": "healthy",
        "service": "ai-studio-backend",
        "environment": settings.ENVIRONMENT,
        "firebaseProjectId": settings.FIREBASE_PROJECT_ID,
        "huggingFaceConfigured": bool(settings.HF_TOKEN)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
