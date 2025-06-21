from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.auth.models import User
from backend.auth.schemas import UserCreateModel
from backend.auth.utils import generate_password_hash


class UserService:
    async def get_user_by_email(self, email: str, session: AsyncSession):
        """
        Fetch a user from the database by email.

        Args:
            email (str): User's email.
            session (AsyncSession): Database session.

        Returns:
            User | None: The user object if found, else None.
        """
        statement = select(User).where(User.email == email)
        result = await session.exec(statement)
        user = result.first()
        return user

    async def user_exists(self, email, session: AsyncSession):
        """
        Check if a user exists with the given email.

        Args:
            email (str): Email to check.
            session (AsyncSession): Database session.

        Returns:
            bool: True if user exists, False otherwise.
        """
        user = await self.get_user_by_email(email, session)
        return True if user is not None else False

    async def create_user(self, user_data: UserCreateModel, session: AsyncSession):
        """
        Create and persist a new user in the database.

        Args:
            user_data (UserCreateModel): Data to create user.
            session (AsyncSession): Database session.

        Returns:
            User: The newly created user instance.
        """
        user_data_dict = user_data.model_dump()
        new_user = User(**user_data_dict)
        new_user.password_hash = generate_password_hash(user_data_dict["password"])
        session.add(new_user)
        await session.commit()
        return new_user
