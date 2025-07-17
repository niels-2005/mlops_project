import pandas as pd
from sklearn.pipeline import Pipeline
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.prediction.models import Predictions
from backend.prediction.schemas import PredictionInput
from backend.prediction.utils import load_threshold

threshold = None


class PredictionService:
    async def get_prediction(
        self,
        pipeline: Pipeline,
        prediction_input: PredictionInput,
        session: AsyncSession,
    ):
        """
        Runs inference on input data using the provided pipeline, applies
        thresholding to determine output class, and persists prediction data.
        """
        input_features = prediction_input.model_dump()
        df = pd.DataFrame([input_features])

        output_proba = pipeline.predict_proba(df)[0][1]

        global threshold
        if threshold is None:
            threshold = load_threshold()
        output = 1 if output_proba > threshold else 0

        output_name = "Heart Disease" if output == 1 else "No Heart Disease"

        await self.save_prediction_data(
            input_features, output_name, output, output_proba, session
        )
        return output, output_proba

    async def save_prediction_data(
        self,
        input_features,
        output_name,
        output,
        output_proba,
        session: AsyncSession,
    ):
        """
        Saves a new prediction record to the database.
        """
        new_prediction = Predictions(
            input_features=input_features,
            output_name=output_name,
            output=output,
            output_proba=output_proba,
        )
        session.add(new_prediction)
        await session.commit()
