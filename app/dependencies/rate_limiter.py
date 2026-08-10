import time
from collections import defaultdict
from fastapi import HTTPException, status

# Sliding window rate limiter state: {user_id: [timestamps]}
REQUEST_HISTORY = defaultdict(list)
WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 15

def enforce_rate_limit(user_id: str) -> None:
    """
    Enforce sliding-window rate limit per user ID.
    Allows up to MAX_REQUESTS_PER_WINDOW requests per WINDOW_SECONDS (e.g. 15 requests / minute).
    """
    now = time.time()
    user_timestamps = REQUEST_HISTORY[user_id]

    # Remove timestamps older than current window
    while user_timestamps and user_timestamps[0] <= now - WINDOW_SECONDS:
        user_timestamps.pop(0)

    if len(user_timestamps) >= MAX_REQUESTS_PER_WINDOW:
        retry_after = int(user_timestamps[0] + WINDOW_SECONDS - now)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: Max {MAX_REQUESTS_PER_WINDOW} generation requests per minute per user. Try again in {retry_after}s.",
            headers={"Retry-After": str(retry_after)}
        )

    user_timestamps.append(now)
