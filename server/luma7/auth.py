"""Bearer token authentication."""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, Query, status


def verify_bearer(authorization: str | None, expected_token: str) -> None:
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing authorization")
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid authorization")
    provided = authorization[len(prefix) :].strip()
    if not hmac.compare_digest(provided, expected_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")


def extract_token(authorization: str | None, token_query: str | None, expected_token: str) -> None:
    if authorization:
        verify_bearer(authorization, expected_token)
        return
    if token_query and hmac.compare_digest(token_query.strip(), expected_token):
        return
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing authorization")


def require_auth(expected_token: str):
    def dependency(
        authorization: str | None = Header(default=None),
        token: str | None = Query(default=None),
    ) -> None:
        extract_token(authorization, token, expected_token)

    return dependency
