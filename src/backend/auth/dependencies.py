from typing import Any, List

from fastapi import Depends, Request, status
from fastapi.exceptions import HTTPException
from fastapi.security import HTTPBearer
from fastapi.security.http import HTTPAuthorizationCredentials
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.auth.models import User
from backend.auth.service import UserService
from backend.db.main import get_session
from backend.db.redis import token_in_blocklist

from .utils import decode_token

user_service = UserService()


class TokenBearer(HTTPBearer):
    def __init__(self, auto_error=True):
        """
        Initialize the TokenBearer with optional auto_error flag.
        """
        super().__init__(auto_error=auto_error)

    async def __call__(self, request: Request) -> HTTPAuthorizationCredentials | None:
        """
        Extract and decode token from request, check blocklist, and verify token data.

        Args:
            request (Request): Incoming HTTP request.

        Returns:
            dict: Decoded token data.

        Raises:
            HTTPException: If token is invalid, expired, or revoked.
        """
        creds = await super().__call__(request)

        token = creds.credentials
        print("token", token)

        token_data = decode_token(token)

        if await token_in_blocklist(token_data["jti"]):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "This token is invalid or has been revoked",
                    "resolution": "Please get new token",
                },
            )

        if not token_data:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "This token is invalid or expired",
                    "resolution": "Please get new token",
                },
            )

        self.verify_token_data(token_data)
        return token_data

    def verify_token_data(self, token_data):
        """
        Abstract method to verify token data. Must be overridden in subclasses.

        Args:
            token_data (dict): Decoded token data.

        Raises:
            NotImplementedError: If not overridden in subclass.
        """
        raise NotImplementedError("Please Override this method in child classes")


class AccessTokenBearer(TokenBearer):
    def verify_token_data(self, token_data: dict) -> None:
        """
        Verify token is not a refresh token.

        Args:
            token_data (dict): Decoded token data.

        Raises:
            HTTPException: If token is a refresh token.
        """
        if token_data and token_data["refresh"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Please provide an access token",
            )


class RefreshTokenBearer(TokenBearer):
    def verify_token_data(self, token_data: dict) -> None:
        """
        Verify token is a refresh token.

        Args:
            token_data (dict): Decoded token data.

        Raises:
            HTTPException: If token is not a refresh token.
        """
        if token_data and not token_data["refresh"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Please provide a refresh token",
            )


async def get_current_user(
    token_details: dict = Depends(AccessTokenBearer()),
    session: AsyncSession = Depends(get_session),
):
    """
    Retrieve current authenticated user from token details.

    Args:
        token_details (dict): Decoded access token data.
        session (AsyncSession): Database session.

    Returns:
        User: User instance fetched by email from the database.
    """
    user_email = token_details["user"]["email"]
    user = await user_service.get_user_by_email(user_email, session)
    return user


class RoleChecker:
    def __init__(self, allowed_roles: List[str]) -> None:
        """
        Initialize RoleChecker with allowed roles.

        Args:
            allowed_roles (List[str]): Roles permitted to access the resource.
        """
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: User = Depends(get_current_user)) -> Any:
        """
        Check if current user's role is allowed.

        Args:
            current_user (User): Current authenticated user.

        Returns:
            bool: True if allowed.

        Raises:
            HTTPException: If user role is not allowed.
        """
        if current_user.role in self.allowed_roles:
            return True

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not allowed to perform this action.",
        )
