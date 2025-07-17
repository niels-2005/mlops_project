import aioredis

from backend.config import Config

JTI_EXPIRY = 3600

token_blocklist = aioredis.StrictRedis(
    host=Config.REDIS_HOST, port=Config.REDIS_PORT, db=0
)


async def add_jti_to_blocklist(jti: str) -> None:
    """
    Add a JWT ID (jti) to the Redis blocklist with expiration to revoke the token.
    """
    await token_blocklist.set(name=jti, value="", ex=JTI_EXPIRY)


async def token_in_blocklist(jti: str) -> bool:
    """
    Check if a JWT ID (jti) is in the Redis blocklist.
    """
    jti = await token_blocklist.get(jti)
    return jti is not None
