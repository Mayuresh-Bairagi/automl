import os
import secrets
from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from dotenv import load_dotenv

load_dotenv()

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Validate API key from request header.

    When AUTOML_API_KEY is set in the environment, all protected endpoints
    require the caller to supply the matching key via the ``X-API-Key`` header.
    If the environment variable is not set, authentication is disabled (useful
    for local development).

    Args:
        api_key: Value extracted from the ``X-API-Key`` request header.

    Returns:
        The validated API key string.

    Raises:
        HTTPException: 401 when the header is missing or 403 when the key is
            invalid.
    """
    expected_key = os.getenv("AUTOML_API_KEY")

    if not expected_key:
        # Auth is disabled – allow all requests (local / dev mode)
        return api_key or ""

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide it via the 'X-API-Key' header.",
        )

    if not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return api_key
