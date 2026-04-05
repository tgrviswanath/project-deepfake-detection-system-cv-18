import asyncio
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.core.detector import analyze
from app.core.validate import validate_image

router = APIRouter(prefix="/api/v1/cv", tags=["deepfake-detection"])


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    validate_image(file, content)
    try:
        return await asyncio.get_running_loop().run_in_executor(None, analyze, content)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")
