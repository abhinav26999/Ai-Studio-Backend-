import logging
from datetime import datetime, timezone
from firebase_functions import firestore_fn
from firebase_functions.options import set_global_options
from firebase_admin import initialize_app, firestore, _apps

set_global_options(region="us-central1")

logger = logging.getLogger("cloud_functions")
INITIAL_BALANCE = 100

def get_db():
    if not _apps:
        initialize_app()
    return firestore.client()

def get_utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

@firestore_fn.on_document_created(document="users/{user_id}")
def on_user_created(event: firestore_fn.Event[firestore_fn.DocumentSnapshot | None]) -> None:
    """
    Firestore Trigger on users/{user_id} creation:
    When a new user document is created in Firestore, automatically provision:
    1. wallets/{userId} with balance = 100.
    2. users/{userId} document with creditBalance = 100.
    3. Log initial welcome bonus in credits_log.
    """
    db = get_db()
    user_id = event.params["user_id"]
    snapshot = event.data

    if not snapshot or not snapshot.exists:
        return

    logger.info(f"Auto-provisioning initial wallet & user doc for new user: {user_id}")

    # 1. Provision wallets/{userId}
    wallet_ref = db.collection("wallets").document(user_id)
    if not wallet_ref.get().exists:
        wallet_ref.set({
            "balance": INITIAL_BALANCE,
            "updatedAt": get_utc_now_iso()
        })

    # 2. Sync creditBalance in users/{userId} doc
    user_ref = db.collection("users").document(user_id)
    user_ref.set({
        "creditBalance": INITIAL_BALANCE,
        "updatedAt": get_utc_now_iso()
    }, merge=True)

    # 3. Log initial welcome bonus transaction
    log_ref = db.collection("credits_log").document()
    log_ref.set({
        "transactionId": log_ref.id,
        "userId": user_id,
        "amount": INITIAL_BALANCE,
        "type": "welcome_bonus",
        "reason": "Initial account signup bonus",
        "timestamp": get_utc_now_iso()
    })

    logger.info(f"Successfully provisioned user {user_id} with {INITIAL_BALANCE} credits.")

@firestore_fn.on_document_written(document="wallets/{user_id}")
def on_wallet_updated(event: firestore_fn.Event[firestore_fn.Change[firestore_fn.DocumentSnapshot | None]]) -> None:
    """
    Firestore Trigger on wallets/{userId}:
    Whenever a wallet balance changes (credit deduction, refund, or admin topup),
    automatically sync the new balance to the creditBalance field in users/{userId}.
    """
    db = get_db()
    user_id = event.params["user_id"]
    new_snapshot = event.data.after

    if not new_snapshot or not new_snapshot.exists:
        logger.warning(f"Wallet doc for {user_id} deleted.")
        return

    wallet_data = new_snapshot.to_dict()
    new_balance = wallet_data.get("balance", 0)

    logger.info(f"Syncing updated wallet balance ({new_balance}) to users/{user_id}")

    user_ref = db.collection("users").document(user_id)
    user_ref.set({
        "creditBalance": new_balance,
        "updatedAt": get_utc_now_iso()
    }, merge=True)
