import re
from typing import Any

EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
CARD_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
PHONE_CANDIDATE_PATTERN = re.compile(r"(?<!\w)\+?\d[\d\s().-]{8,}\d(?!\w)")
SECRET_PATTERNS = (
    (
        "named_secret",
        re.compile(
            r"(?i)\b(?:api[_ -]?key|access[_ -]?token|secret|password)\b"
            r"\s*[:=]\s*[\"']?[A-Za-z0-9_./+\-=]{8,}[\"']?"
        ),
    ),
    (
        "provider_api_key",
        re.compile(r"\b(?:gsk_|sk-|ghp_)[A-Za-z0-9_-]{16,}\b", re.IGNORECASE),
    ),
    ("aws_access_key", re.compile(r"\bAKIA[A-Z0-9]{16}\b")),
)
INJECTION_PATTERNS = (
    (
        "ignore_previous_instructions",
        re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
    ),
    (
        "reveal_system_prompt",
        re.compile(r"reveal\s+(?:the\s+)?system\s+prompt", re.IGNORECASE),
    ),
    (
        "show_hidden_instructions",
        re.compile(r"show\s+(?:the\s+)?hidden\s+instructions", re.IGNORECASE),
    ),
    (
        "bypass_safety_rules",
        re.compile(r"bypass\s+(?:the\s+)?safety\s+rules", re.IGNORECASE),
    ),
    (
        "unrestricted_assistant",
        re.compile(r"act\s+as\s+an?\s+unrestricted\s+assistant", re.IGNORECASE),
    ),
    (
        "disregard_instructions",
        re.compile(r"disregard\s+(?:all\s+|your\s+)?instructions", re.IGNORECASE),
    ),
)


def _looks_like_phone(candidate: str) -> bool:
    digit_count = sum(character.isdigit() for character in candidate)
    return 10 <= digit_count <= 15


def _redact_phones(text: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        return "[REDACTED_PHONE]" if _looks_like_phone(match.group()) else match.group()

    return PHONE_CANDIDATE_PATTERN.sub(replacement, text)


def _contains_phone(text: str) -> bool:
    return any(
        _looks_like_phone(match.group())
        for match in PHONE_CANDIDATE_PATTERN.finditer(text)
    )


def _apply_redaction(text: str) -> str:
    sanitized = text
    for _, pattern in SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED_SECRET]", sanitized)
    sanitized = CARD_PATTERN.sub("[REDACTED_CARD]", sanitized)
    sanitized = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", sanitized)
    return _redact_phones(sanitized)


def inspect_and_sanitize(
    query: str, answer: str, contexts: list[str]
) -> dict[str, Any]:
    """Apply baseline regex checks; false positives and bypasses remain possible."""
    joined = "\n".join([query, answer, *contexts])
    categories: list[str] = []
    matched_rules: list[str] = []

    if EMAIL_PATTERN.search(joined):
        categories.append("email")
        matched_rules.append("email_address")
    if _contains_phone(joined):
        categories.append("phone")
        matched_rules.append("phone_number")
    if CARD_PATTERN.search(joined):
        categories.append("credit_card")
        matched_rules.append("credit_card_like_number")

    for rule_name, pattern in SECRET_PATTERNS:
        if pattern.search(joined):
            if "secret" not in categories:
                categories.append("secret")
            matched_rules.append(rule_name)

    for rule_name, pattern in INJECTION_PATTERNS:
        if pattern.search(joined):
            if "prompt_injection" not in categories:
                categories.append("prompt_injection")
            matched_rules.append(rule_name)

    return {
        "pii_detected": any(
            category in categories for category in ("email", "phone", "credit_card")
        ),
        "secret_detected": "secret" in categories,
        "prompt_injection_detected": "prompt_injection" in categories,
        "categories": categories,
        "matched_rules": matched_rules,
        "sanitized_query": _apply_redaction(query),
        "sanitized_answer": _apply_redaction(answer),
        "sanitized_contexts": [_apply_redaction(context) for context in contexts],
    }
