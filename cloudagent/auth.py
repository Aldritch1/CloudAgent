import logging

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from cloudagent.config import settings

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    if settings.jwt_disabled:
        return "anonymous"

    secret = settings.jwt_secret.get_secret_value()
    if not secret:
        logger.warning("JWT secret not configured, auth disabled")
        return "anonymous"

    if not token:
        raise HTTPException(status_code=401, detail="未提供认证令牌")

    try:
        payload = jwt.decode(token, secret, algorithms=[settings.jwt_algorithm])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="无效的认证令牌")
        return user_id
    except (JWTError, Exception):
        raise HTTPException(status_code=401, detail="认证令牌无效或已过期")
