from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class WalletDoc(BaseModel):
    userId: str
    balance: int = 100
    updatedAt: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

class CreditTransaction(BaseModel):
    transactionId: Optional[str] = None
    userId: str
    amount: int
    type: str  # "deduction", "refund", "welcome_bonus", "admin_topup"
    jobId: Optional[str] = None
    reason: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
