"""SMTP email delivery.

Sends the finished briefing as a plain-text email, optionally paired
with an HTML alternative (see brief.build_html_briefing — built
directly from the same code-generated flag data, never from LLM text,
so it carries the same safety guarantee as the plain-text fallback).
No retry logic here beyond what smtplib gives naturally — by the time
this runs, the briefing text has already been produced and logged (see
main.py), so a delivery failure means "the email didn't arrive," never
"the briefing was lost."
"""

from __future__ import annotations

import smtplib
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Callable, Optional, Union


class DeliveryError(Exception):
    """SMTP send failed. Callers should still have the briefing text
    available some other way (stdout, logs/) — this is a delivery
    failure, not a run failure."""


@dataclass(frozen=True)
class SMTPConfig:
    host: str
    port: int
    use_tls: bool
    username: str
    password: str
    from_address: str
    to_address: str
    timeout_seconds: float = 15.0


def build_message(subject: str, body: str, config: SMTPConfig, html_body: Optional[str] = None) -> Union[MIMEText, MIMEMultipart]:
    """Plain-text-only message when `html_body` is omitted (matches
    every existing caller/test). With `html_body`, builds a
    multipart/alternative message — plain part first, HTML part last,
    per RFC 2046 ("last part is the most preferred"), since most mail
    clients render the last alternative they understand."""
    if html_body is None:
        msg = MIMEText(body, "plain", "utf-8")
    else:
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText(body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html", "utf-8"))
    msg["Subject"] = subject
    msg["From"] = config.from_address
    msg["To"] = config.to_address
    return msg


def send_briefing(
    subject: str,
    body: str,
    config: SMTPConfig,
    smtp_client_factory: Optional[Callable[[], smtplib.SMTP]] = None,
    html_body: Optional[str] = None,
) -> None:
    """`smtp_client_factory` is injectable for testing — must return a
    context-manager-capable SMTP-like object (smtplib.SMTP and
    smtplib.SMTP_SSL both qualify). Defaults to a real SMTP connection
    to config.host/port."""
    msg = build_message(subject, body, config, html_body=html_body)
    factory = smtp_client_factory or (lambda: smtplib.SMTP(config.host, config.port, timeout=config.timeout_seconds))

    try:
        with factory() as server:
            if config.use_tls:
                server.starttls()
            if config.username and config.password:
                server.login(config.username, config.password)
            server.sendmail(config.from_address, [config.to_address], msg.as_string())
    except (smtplib.SMTPException, OSError) as exc:
        raise DeliveryError(f"failed to send briefing email: {exc}") from exc


def load_smtp_config(cfg: dict, env: dict) -> SMTPConfig:
    """Merges config.yaml's delivery section with .env overrides (env
    wins, since addresses/credentials are secrets that shouldn't live
    in the checked-in config file)."""
    delivery = cfg["delivery"]
    from_address = env.get("BRIEFING_FROM_ADDRESS") or delivery.get("from_address") or ""
    to_address = env.get("BRIEFING_TO_ADDRESS") or delivery.get("to_address") or ""
    if not from_address or not to_address:
        raise DeliveryError(
            "delivery from/to address not configured -- set delivery.from_address/to_address in "
            "config.yaml or BRIEFING_FROM_ADDRESS/BRIEFING_TO_ADDRESS in .env"
        )
    return SMTPConfig(
        host=delivery["smtp_host"],
        port=delivery["smtp_port"],
        use_tls=delivery["smtp_use_tls"],
        username=env.get("BRIEFING_SMTP_USERNAME") or from_address,
        password=env.get("BRIEFING_SMTP_PASSWORD", ""),
        from_address=from_address,
        to_address=to_address,
    )
