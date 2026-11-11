"""Small, dependency-free RePro v8 algorithm reference; not a full trainer."""

from .graphs import ColoredGraph, CanonicalForm, TransitionKey
from .graphs import canonicalize, canonical_transition
from .core import BudgetBuckets, SupportPolicy, ValueEstimate, ContinuationTable
from .core import Outcome, SUCCESS, FAILURE, OTHER, KernelRow, RecoveryCounts
from .core import RecoveryValues, solve_recovery_values
from .advantages import supported_gae, af_group_mask, combine_advantages
from .advantages import action_to_token_advantages

__all__ = [
    "ColoredGraph", "CanonicalForm", "TransitionKey", "canonicalize",
    "canonical_transition", "BudgetBuckets", "SupportPolicy", "ValueEstimate",
    "ContinuationTable", "Outcome", "SUCCESS", "FAILURE", "OTHER", "KernelRow",
    "RecoveryCounts", "RecoveryValues", "solve_recovery_values", "supported_gae",
    "af_group_mask", "combine_advantages", "action_to_token_advantages",
]
