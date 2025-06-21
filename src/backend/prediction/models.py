import uuid
from datetime import date, datetime
from typing import Dict, Optional

import sqlalchemy.dialects.postgresql as pg
from sqlmodel import Column, Field, Relationship, SQLModel


class Predictions(SQLModel, table=True):
    """
    SQLModel ORM class representing stored prediction records.

    Attributes:
        uid (UUID): Unique identifier for each prediction (primary key).
        input_features (Dict): JSONB column storing input features.
        output_name (str): Name/label of prediction output.
        output (int): Predicted class label.
        output_proba (float): Probability of the positive class.
        time (datetime): Timestamp when prediction was made.
    """

    __tablename__ = "predictions"

    uid: uuid.UUID = Field(
        sa_column=Column(
            pg.UUID, primary_key=True, unique=True, nullable=False, default=uuid.uuid4
        )
    )
    input_features: Dict = Field(sa_column=Column(pg.JSONB, nullable=False))
    output_name: str = Field(nullable=False)
    output: int = Field(nullable=False)
    output_proba: float = Field(nullable=False)
    time: datetime = Field(sa_column=Column(pg.TIMESTAMP, default=datetime.now))
