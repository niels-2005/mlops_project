import pandas as pd
from sklearn.pipeline import Pipeline
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.prediction.models import Predictions
from backend.prediction.schemas import PredictionInput


class PredictionService:
    async def get_prediction(
        self,
        pipeline: Pipeline,
        prediction_input: PredictionInput,
        session: AsyncSession,
    ):
        input_features = prediction_input.model_dump()
        df = pd.DataFrame([input_features])

        output = int(pipeline.predict(df)[0])
        output_proba = pipeline.predict_proba(df)[0][1]

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
        new_prediction = Predictions(
            input_features=input_features,
            output_name=output_name,
            output=output,
            output_proba=output_proba,
        )
        session.add(new_prediction)
        await session.commit()
