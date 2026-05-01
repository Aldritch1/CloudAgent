import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest
from fastapi import HTTPException

from cloudagent.auth import get_current_user


@patch("cloudagent.auth.settings")
def test_valid_token_returns_user_id(mock_settings):
    mock_settings.jwt_disabled = False
    mock_settings.jwt_secret.get_secret_value.return_value = "secret"
    mock_settings.jwt_algorithm = "HS256"

    with patch("cloudagent.auth.jwt.decode") as mock_decode:
        mock_decode.return_value = {"sub": "user-123"}
        result = get_current_user("Bearer valid-token")

    assert result == "user-123"


@patch("cloudagent.auth.settings")
def test_disabled_auth_allows_all(mock_settings):
    mock_settings.jwt_disabled = True

    result = get_current_user(None)
    assert result == "anonymous"


@patch("cloudagent.auth.settings")
def test_missing_secret_allows_all(mock_settings):
    mock_settings.jwt_disabled = False
    mock_settings.jwt_secret.get_secret_value.return_value = ""

    result = get_current_user(None)
    assert result == "anonymous"


@patch("cloudagent.auth.settings")
def test_missing_token_raises_401(mock_settings):
    mock_settings.jwt_disabled = False
    mock_settings.jwt_secret.get_secret_value.return_value = "secret"
    mock_settings.jwt_algorithm = "HS256"

    with pytest.raises(HTTPException) as exc_info:
        get_current_user(None)

    assert exc_info.value.status_code == 401


@patch("cloudagent.auth.settings")
def test_invalid_token_raises_401(mock_settings):
    mock_settings.jwt_disabled = False
    mock_settings.jwt_secret.get_secret_value.return_value = "secret"
    mock_settings.jwt_algorithm = "HS256"

    with patch("cloudagent.auth.jwt.decode", side_effect=Exception("bad token")):
        with pytest.raises(HTTPException) as exc_info:
            get_current_user("Bearer invalid-token")

    assert exc_info.value.status_code == 401
