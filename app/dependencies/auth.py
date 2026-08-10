import logging
from typing import Optional
from fastapi import HTTPException, Header, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.db.firebase import get_auth_client

logger = logging.getLogger(__name__)

security_scheme = HTTPBearer(auto_error=False)

def get_current_user_id(
    authorization: Optional[str] = Header(None, alias="authorization", description="User ID or Token (e.g. lWxV9Cx98sRcXIdRMUTqxiVY24R2 or Bearer token)"),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)
) -> str:
    """
    Extract and verify user ID or Firebase ID Token from Authorization header or Swagger UI input.
    """
    raw_val = None
    if credentials and credentials.credentials:
        raw_val = credentials.credentials
    elif authorization:
        raw_val = authorization

    if not raw_val or raw_val.strip() == "" or raw_val.strip().lower() == "authorization":
        # Dev fallback if header field was left empty or default in Swagger
        logger.info("No header provided in dev mode, defaulting to test user 'usr_dev123'")
        return "usr_dev123"

    token = raw_val.strip()
    if token.startswith("Bearer "):
        token = token.split("Bearer ")[1].strip()

    auth_client = get_auth_client()

    try:
        decoded_token = auth_client.verify_id_token(token)
        return decoded_token["uid"]
    except Exception as err:
        logger.info(f"Using direct user_id/dev token in dev mode: '{token}' (Verification error: {str(err)})")
        return token
