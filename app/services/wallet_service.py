import logging
from datetime import datetime, timezone
from google.cloud import firestore
from fastapi import HTTPException
from app.db.firebase import get_db

logger = logging.getLogger(__name__)

INITIAL_BALANCE = 100

def get_utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def auto_provision_wallet(user_id: str) -> dict:
    """Auto-provision wallets/{userId} = { balance: 100, updatedAt: now } if not present."""
    db = get_db()
    wallet_ref = db.collection("wallets").document(user_id)
    doc = wallet_ref.get()
    
    if not doc.exists:
        data = {
            "balance": INITIAL_BALANCE,
            "updatedAt": get_utc_now_iso()
        }
        wallet_ref.set(data)
        
        # Log initial welcome bonus
        log_ref = db.collection("credits_log").document()
        log_ref.set({
            "transactionId": log_ref.id,
            "userId": user_id,
            "amount": INITIAL_BALANCE,
            "type": "welcome_bonus",
            "reason": "Initial account provisioning",
            "timestamp": get_utc_now_iso()
        })
        logger.info(f"Provisioned wallet for user {user_id} with initial balance {INITIAL_BALANCE}")
        return data
    return doc.to_dict()

def get_wallet_balance(user_id: str) -> dict:
    """Retrieve wallet for user, provisioning if missing."""
    db = get_db()
    wallet_ref = db.collection("wallets").document(user_id)
    doc = wallet_ref.get()
    if not doc.exists:
        return auto_provision_wallet(user_id)
    return doc.to_dict()

@firestore.transactional
def _transactional_deduct(transaction, wallet_ref, user_id: str, cost: int, job_id: str):
    snapshot = wallet_ref.get(transaction=transaction)
    if not snapshot.exists:
        current_balance = INITIAL_BALANCE
        # Set initial if doc missing
        transaction.set(wallet_ref, {
            "balance": current_balance,
            "updatedAt": get_utc_now_iso()
        })
    else:
        wallet_data = snapshot.to_dict()
        current_balance = wallet_data.get("balance", 0)

    if current_balance < cost:
        raise ValueError(f"Insufficient credits: current balance is {current_balance}, required is {cost}")

    new_balance = current_balance - cost
    transaction.update(wallet_ref, {
        "balance": new_balance,
        "updatedAt": get_utc_now_iso()
    })
    return new_balance

def check_and_deduct_credits(user_id: str, cost: int, job_id: str) -> int:
    """
    Race-safe transactional credit deduction.
    Throws HTTPException 402 (Payment Required) if balance is insufficient.
    """
    db = get_db()
    wallet_ref = db.collection("wallets").document(user_id)
    transaction = db.transaction()

    try:
        new_balance = _transactional_deduct(transaction, wallet_ref, user_id, cost, job_id)
        
        # Log deduction in credits_log
        log_ref = db.collection("credits_log").document()
        log_ref.set({
            "transactionId": log_ref.id,
            "userId": user_id,
            "amount": -cost,
            "type": "deduction",
            "jobId": job_id,
            "timestamp": get_utc_now_iso()
        })
        logger.info(f"Deducted {cost} credits from user {user_id} for job {job_id}. New balance: {new_balance}")
        return new_balance
    except ValueError as err:
        logger.warning(f"Credit deduction failed for user {user_id}: {str(err)}")
        raise HTTPException(status_code=402, detail=str(err))
    except Exception as err:
        logger.error(f"Unexpected error during credit deduction for user {user_id}: {str(err)}")
        raise HTTPException(status_code=500, detail="Transaction error during credit deduction")

@firestore.transactional
def _transactional_refund(transaction, wallet_ref, cost: int):
    snapshot = wallet_ref.get(transaction=transaction)
    current_balance = snapshot.to_dict().get("balance", 0) if snapshot.exists else 0
    new_balance = current_balance + cost
    transaction.set(wallet_ref, {
        "balance": new_balance,
        "updatedAt": get_utc_now_iso()
    }, merge=True)
    return new_balance

def refund_credits(user_id: str, cost: int, job_id: str, reason: str = "GPU execution error") -> int:
    """Refund deducted credits back to user's wallet when job fails."""
    db = get_db()
    wallet_ref = db.collection("wallets").document(user_id)
    transaction = db.transaction()

    try:
        new_balance = _transactional_refund(transaction, wallet_ref, cost)
        
        # Log refund
        log_ref = db.collection("credits_log").document()
        log_ref.set({
            "transactionId": log_ref.id,
            "userId": user_id,
            "amount": cost,
            "type": "refund",
            "jobId": job_id,
            "reason": reason,
            "timestamp": get_utc_now_iso()
        })
        logger.info(f"Refunded {cost} credits to user {user_id} for job {job_id}. New balance: {new_balance}")
        return new_balance
    except Exception as err:
        logger.error(f"Failed to refund credits for user {user_id}: {str(err)}")
        raise err
