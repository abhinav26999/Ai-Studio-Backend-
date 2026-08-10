import logging
from fastapi import APIRouter, Depends, status
from app.services.wallet_service import get_wallet_balance
from app.dependencies.auth import get_current_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/wallet", tags=["wallet"])

@router.get("/{user_id}", status_code=status.HTTP_200_OK)
async def get_wallet(user_id: str):
    """
    Fetch live balance for user wallet (wallets/{userId}).
    Auto-provisions with 100 free credits if wallet does not exist yet.
    """
    wallet_data = get_wallet_balance(user_id)
    return {
        "userId": user_id,
        "balance": wallet_data.get("balance", 100),
        "updatedAt": wallet_data.get("updatedAt")
    }
