import smtplib
from unittest.mock import MagicMock

import pytest

from deliver import DeliveryError, SMTPConfig, build_message, load_smtp_config, send_briefing

CONFIG = SMTPConfig(
    host="smtp.gmail.com",
    port=587,
    use_tls=True,
    username="me@gmail.com",
    password="app-password",
    from_address="me@gmail.com",
    to_address="me@gmail.com",
)


def test_build_message_sets_headers_and_body():
    msg = build_message("Morning Briefing", "AAPL gap +2.0%", CONFIG)
    assert msg["Subject"] == "Morning Briefing"
    assert msg["From"] == "me@gmail.com"
    assert msg["To"] == "me@gmail.com"
    assert msg.get_payload(decode=True).decode("utf-8").strip() == "AAPL gap +2.0%"


def _fake_smtp_factory():
    server = MagicMock()
    server.__enter__.return_value = server
    server.__exit__.return_value = False
    factory = MagicMock(return_value=server)
    return factory, server


def test_send_briefing_happy_path_calls_starttls_login_sendmail():
    factory, server = _fake_smtp_factory()
    send_briefing("Subject", "Body", CONFIG, smtp_client_factory=factory)
    server.starttls.assert_called_once()
    server.login.assert_called_once_with("me@gmail.com", "app-password")
    assert server.sendmail.call_count == 1
    args = server.sendmail.call_args[0]
    assert args[0] == "me@gmail.com"
    assert args[1] == ["me@gmail.com"]


def test_send_briefing_skips_starttls_when_disabled():
    factory, server = _fake_smtp_factory()
    config = SMTPConfig(**{**CONFIG.__dict__, "use_tls": False})
    send_briefing("Subject", "Body", config, smtp_client_factory=factory)
    server.starttls.assert_not_called()


def test_send_briefing_skips_login_without_credentials():
    factory, server = _fake_smtp_factory()
    config = SMTPConfig(**{**CONFIG.__dict__, "username": "", "password": ""})
    send_briefing("Subject", "Body", config, smtp_client_factory=factory)
    server.login.assert_not_called()


def test_send_briefing_wraps_smtp_exception():
    factory, server = _fake_smtp_factory()
    server.sendmail.side_effect = smtplib.SMTPAuthenticationError(535, b"bad creds")
    with pytest.raises(DeliveryError):
        send_briefing("Subject", "Body", CONFIG, smtp_client_factory=factory)


def test_send_briefing_wraps_os_error():
    factory, server = _fake_smtp_factory()
    server.sendmail.side_effect = OSError("connection refused")
    with pytest.raises(DeliveryError):
        send_briefing("Subject", "Body", CONFIG, smtp_client_factory=factory)


# ---------------------------------------------------------------------------
# load_smtp_config
# ---------------------------------------------------------------------------

CFG = {"delivery": {"smtp_host": "smtp.gmail.com", "smtp_port": 587, "smtp_use_tls": True, "from_address": "", "to_address": ""}}


def test_load_smtp_config_from_env():
    env = {
        "BRIEFING_FROM_ADDRESS": "me@gmail.com",
        "BRIEFING_TO_ADDRESS": "me@gmail.com",
        "BRIEFING_SMTP_USERNAME": "me@gmail.com",
        "BRIEFING_SMTP_PASSWORD": "secret",
    }
    result = load_smtp_config(CFG, env)
    assert result.from_address == "me@gmail.com"
    assert result.password == "secret"


def test_load_smtp_config_username_defaults_to_from_address():
    env = {"BRIEFING_FROM_ADDRESS": "me@gmail.com", "BRIEFING_TO_ADDRESS": "me@gmail.com"}
    result = load_smtp_config(CFG, env)
    assert result.username == "me@gmail.com"


def test_load_smtp_config_prefers_env_over_config_yaml():
    cfg = {"delivery": dict(CFG["delivery"], from_address="config@example.com", to_address="config@example.com")}
    env = {"BRIEFING_FROM_ADDRESS": "env@example.com", "BRIEFING_TO_ADDRESS": "env@example.com"}
    result = load_smtp_config(cfg, env)
    assert result.from_address == "env@example.com"


def test_load_smtp_config_raises_when_addresses_missing():
    with pytest.raises(DeliveryError):
        load_smtp_config(CFG, {})
