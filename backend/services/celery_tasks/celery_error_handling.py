"""
Shared exception handling for Celery tasks so result backend never receives
non-serializable exceptions (e.g. billiard.exceptions.SoftTimeLimitExceeded).
"""

from celery.exceptions import SoftTimeLimitExceeded

# Catch both celery and billiard SoftTimeLimitExceeded so the exception never
# reaches the result backend (which expects exc_type and can raise ValueError).
SOFT_TIME_LIMIT_EXCEEDED_TYPES = (SoftTimeLimitExceeded,)
try:
    from billiard.exceptions import SoftTimeLimitExceeded as BilliardSoftTimeLimitExceeded
    SOFT_TIME_LIMIT_EXCEEDED_TYPES = (SoftTimeLimitExceeded, BilliardSoftTimeLimitExceeded)
except ImportError:
    pass
