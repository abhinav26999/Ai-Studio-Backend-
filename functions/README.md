# Firebase Cloud Functions (Python)

This directory contains production Cloud Functions for **AI Studio Backend**:

1. **`on_user_created` (Auth Trigger):**
   - Triggers automatically when a user registers via Firebase Auth.
   - Auto-provisions `wallets/{userId}` with 100 free credits.
   - Creates/merges `users/{userId}` with `creditBalance: 100`.
   - Records a welcome bonus transaction in `credits_log`.

2. **`on_wallet_updated` (Firestore Trigger):**
   - Listens to changes on `wallets/{user_id}`.
   - Automatically syncs the updated credit `balance` to `users/{userId}.creditBalance`.

---

## 🚀 Deployment Instructions

### Prerequisites
Make sure Firebase CLI is installed and logged in:
```bash
npm install -g firebase-tools
firebase login
```

### Deploy Functions to Firebase Project (`ai-studio-637ab`)
Run from project root:
```bash
firebase deploy --only functions
```
