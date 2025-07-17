import logging
import uuid
from datetime import datetime, timedelta

import jwt
from passlib.context import CryptContext

from backend.config import Config

passwd_context = CryptContext(schemes=["bcrypt"])


def generate_password_hash(password: str) -> str:
    """
    Generate a secure hash for a given password.
    """
    hash = passwd_context.hash(password)
    return hash


def verify_password(password: str, hash: str) -> bool:
    """
    Verify a plain password against its hashed version.
    """
    return passwd_context.verify(password, hash)


def create_access_token(
    user_data: dict, expiry: timedelta = None, refresh: bool = False
) -> str:
    """
    Create a JWT access or refresh token.
    """
    payload = {
        "user": user_data,
        "exp": datetime.now()
        + (expiry if expiry is not None else timedelta(minutes=60)),
        "jti": str(uuid.uuid4()),
        "refresh": refresh,
    }

    token = jwt.encode(
        payload=payload, key=Config.JWT_SECRET, algorithm=Config.JWT_ALGORITHM
    )
    return token


def decode_token(token: str) -> dict:
    """
    Decode and verify a JWT token.
    """
    try:
        token_data = jwt.decode(
            jwt=token, algorithms=[Config.JWT_ALGORITHM], key=Config.JWT_SECRET
        )
        return token_data
    except jwt.PyJWTError as jwte:
        logging.exception(jwte)
        return None
    except Exception as e:
        logging.exception(e)
        return None
