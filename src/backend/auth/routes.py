from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, status
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.auth.dependencies import (
    AccessTokenBearer,
    RefreshTokenBearer,
    RoleChecker,
    get_current_user,
)
from backend.auth.schemas import UserCreateModel, UserLoginModel, UserModel
from backend.auth.service import UserService
from backend.auth.utils import create_access_token, verify_password
from backend.db.main import get_session
from backend.db.redis import add_jti_to_blocklist

auth_router = APIRouter()
user_service = UserService()
role_checker = RoleChecker(["admin", "user"])

REFRESH_TOKEN_EXPIRY = 2


@auth_router.post(
    "/signup", response_model=UserModel, status_code=status.HTTP_201_CREATED
)
async def create_user_account(
    user_data: UserCreateModel, session: AsyncSession = Depends(get_session)
):
    """
    Register a new user account.

    Args:
        user_data (UserCreateModel): Data required to create a user.
        session (AsyncSession): Database session dependency.

    Raises:
        HTTPException: If user with the given email already exists.

    Returns:
        UserModel: The created user.
    """
    email = user_data.email

    user_exists = await user_service.user_exists(email, session)
    if user_exists:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User with email already exists",
        )

    new_user = await user_service.create_user(user_data, session)
    return new_user


@auth_router.post("/login")
async def login_users(
    login_data: UserLoginModel, session: AsyncSession = Depends(get_session)
):
    """
    Authenticate user and return access and refresh tokens on success.

    Args:
        login_data (UserLoginModel): User login credentials.
        session (AsyncSession): Database session dependency.

    Raises:
        HTTPException: If email or password is invalid.

    Returns:
        JSONResponse: Contains access token, refresh token, and user info.
    """
    email = login_data.email
    password = login_data.password

    user = await user_service.get_user_by_email(email, session)

    if user is not None:
        password_valid = verify_password(password, user.password_hash)

        if password_valid:
            access_token = create_access_token(
                user_data={"email": user.email, "user_uid": str(user.uid)}
            )

            refresh_token = create_access_token(
                user_data={"email": user.email, "user_uid": str(user.uid)},
                refresh=True,
                expiry=timedelta(days=REFRESH_TOKEN_EXPIRY),
            )

            return JSONResponse(
                content={
                    "message": "Login successful",
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "user": {
                        "username": user.username,
                        "uid": str(user.uid),
                    },
                }
            )

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN, detail="Invalid Email Or Password"
    )


@auth_router.get("/logout")
async def revoke_token(token_details: dict = Depends(AccessTokenBearer())):
    """
    Revoke (invalidate) the current access token by adding its JTI to the blocklist.

    Args:
        token_details (dict): Decoded access token details injected by dependency.

    Returns:
        JSONResponse: Confirmation message of logout success.
    """
    jti = token_details["jti"]

    await add_jti_to_blocklist(jti)

    return JSONResponse(
        content={"message": "Logged Out Successfully"}, status_code=status.HTTP_200_OK
    )


@auth_router.get("/refresh_token")
async def get_new_access_token(token_details: dict = Depends(RefreshTokenBearer())):
    """
    Generate a new access token using a valid refresh token.

    Args:
        token_details (dict): Decoded refresh token details injected by dependency.

    Raises:
        HTTPException: If the refresh token is expired or invalid.

    Returns:
        JSONResponse: Contains the new access token.
    """
    expiry_timestamp = token_details["exp"]

    if datetime.fromtimestamp(expiry_timestamp) > datetime.now():
        new_access_token = create_access_token(user_data=token_details["user"])

        return JSONResponse(content={"access_token": new_access_token})

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Or expired token"
    )


@auth_router.get("/me", response_model=UserModel)
async def get_current_user(
    user=Depends(get_current_user), role_checker: bool = Depends(role_checker)
):
    """
    Retrieve the current authenticated user's information.

    Args:
        user (User): Current user injected by dependency.
        role_checker (bool): Role-based access control check.

    Returns:
        UserModel: The current authenticated user.
    """
    return user
