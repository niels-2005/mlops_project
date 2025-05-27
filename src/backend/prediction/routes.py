from fastapi import APIRouter, Depends
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.db.main import get_session
from backend.prediction.schemas import PredictionInput, PredictionResponseData
from backend.prediction.service import PredictionService
from backend.prediction.utils import load_pipeline

pipeline = load_pipeline()

prediction_router = APIRouter()
prediction_service = PredictionService()


@prediction_router.post("/", response_model=PredictionResponseData)
async def make_prediction(
    prediction_input: PredictionInput, session: AsyncSession = Depends(get_session)
) -> PredictionResponseData:
    output, output_proba = await prediction_service.get_prediction(
        pipeline, prediction_input, session
    )
    return PredictionResponseData(output=output, output_proba=output_proba)
