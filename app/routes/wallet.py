import logging
from fastapi import APIRouter, Depends, HTTPException, status
from app.services.wallet_service import get_wallet_balance
from app.dependencies.auth import get_current_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/wallet", tags=["wallet"])

@router.get("/{user_id}", status_code=status.HTTP_200_OK)
async def get_wallet(
    user_id: str,
    authenticated_user_id: str = Depends(get_current_user_id)
):
    """
    Fetch live balance for user wallet (wallets/{userId}).
    Requires authorization header. Rejects request if authenticated user ID does not match target user_id.
    Auto-provisions with 100 free credits if wallet does not exist yet.
    """
    if authenticated_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Forbidden: Cannot query wallet balance for another user ('{user_id}')"
        )

    wallet_data = get_wallet_balance(user_id)
    return {
        "userId": user_id,
        "balance": wallet_data.get("balance", 100),
        "updatedAt": wallet_data.get("updatedAt")
    }
