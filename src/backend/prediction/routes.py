from fastapi import APIRouter, Depends, status
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.auth.dependencies import AccessTokenBearer, RoleChecker
from backend.db.main import get_session
from backend.prediction.schemas import PredictionInput, PredictionResponseData
from backend.prediction.service import PredictionService
from backend.prediction.utils import load_pipeline

prediction_router = APIRouter()
prediction_service = PredictionService()

role_checker = RoleChecker(["admin", "user"])

prediction_pipeline = None


@prediction_router.post(
    "/", response_model=PredictionResponseData, status_code=status.HTTP_200_OK
)
async def make_prediction(
    prediction_input: PredictionInput,
    session: AsyncSession = Depends(get_session),
    role_checker: bool = Depends(role_checker),
    token_details: dict = Depends(AccessTokenBearer()),
) -> PredictionResponseData:
    """
    Handles prediction requests by loading the pipeline if necessary,
    running inference, and returning prediction results.

    Args:
        prediction_input (PredictionInput): Input data for prediction.
        session (AsyncSession): Database session for saving prediction.
        role_checker (bool): Authorization dependency.
        token_details (dict): Authentication token info.

    Returns:
        PredictionResponseData: Contains prediction label and probability.

    Raises:
        Exception: Propagates exceptions during prediction or DB operations.
    """
    global prediction_pipeline
    if prediction_pipeline is None:
        prediction_pipeline = load_pipeline()
    output, output_proba = await prediction_service.get_prediction(
        prediction_pipeline, prediction_input, session
    )
    return PredictionResponseData(output=output, output_proba=output_proba)
