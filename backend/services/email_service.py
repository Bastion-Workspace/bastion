"""
Email Service - Handles sending emails via SMTP
Supports multiple backends: Mailpit, Postfix, SMTP relay services
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List
from config import settings

logger = logging.getLogger(__name__)


class EmailService:
    """Service for sending emails via SMTP"""
    
    def __init__(self):
        self.smtp_host = settings.SMTP_HOST
        self.smtp_port = settings.SMTP_PORT
        self.smtp_user = settings.SMTP_USER
        self.smtp_password = settings.SMTP_PASSWORD
        self.smtp_use_tls = settings.SMTP_USE_TLS
        self.from_email = settings.SMTP_FROM_EMAIL
        self.from_name = settings.SMTP_FROM_NAME
        self.enabled = settings.SMTP_ENABLED
    
    async def send_email(
        self,
        to_email: str,
        subject: str,
        body_text: str,
        body_html: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        from_email: Optional[str] = None,
        from_name: Optional[str] = None,
        smtp_config: Optional[dict] = None,
    ) -> bool:
        """
        Send an email via SMTP. Uses smtp_config when provided (e.g. from DB), else instance config (env).
        """
        if smtp_config:
            enabled = bool(
                smtp_config.get("host")
                and (smtp_config.get("from_email") or smtp_config.get("user"))
            )
            host = smtp_config.get("host", "")
            port = int(smtp_config.get("port", 587))
            user = smtp_config.get("user", "")
            password = smtp_config.get("password", "")
            use_tls = smtp_config.get("use_tls", True)
            sender_email = from_email or smtp_config.get("from_email", "")
            sender_name = from_name or smtp_config.get("from_name", "")
        else:
            enabled = self.enabled
            host = self.smtp_host
            port = self.smtp_port
            user = self.smtp_user
            password = self.smtp_password
            use_tls = self.smtp_use_tls
            sender_email = from_email or self.from_email
            sender_name = from_name or self.from_name

        if not enabled or not host:
            logger.warning("Email sending is disabled or SMTP host not set")
            return False

        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{sender_name} <{sender_email}>" if sender_name else sender_email
            msg['To'] = to_email

            if cc:
                msg['Cc'] = ', '.join(cc)

            text_part = MIMEText(body_text, 'plain')
            msg.attach(text_part)
            if body_html:
                html_part = MIMEText(body_html, 'html')
                msg.attach(html_part)

            recipients = [to_email]
            if cc:
                recipients.extend(cc)
            if bcc:
                recipients.extend(bcc)

            with smtplib.SMTP(host, port) as server:
                if use_tls:
                    server.starttls()
                if user and password:
                    server.login(user, password)
                server.send_message(msg, to_addrs=recipients)

            logger.info("Email sent successfully to %s", to_email)
            return True

        except Exception as e:
            logger.error("Failed to send email to %s: %s", to_email, e)
            return False
    
    async def send_password_reset_email(
        self,
        to_email: str,
        username: str,
        reset_token: str,
        reset_url: str
    ) -> bool:
        """Send password reset email"""
        subject = "Password Reset Request - Bastion AI Workspace"
        
        body_text = f"""
Hello {username},

You requested a password reset for your Bastion AI Workspace account.

Click the following link to reset your password:
{reset_url}

This link will expire in 1 hour.

If you did not request this password reset, please ignore this email.

Best regards,
Bastion AI Workspace
"""
        
        body_html = f"""
<html>
<body>
    <h2>Password Reset Request</h2>
    <p>Hello {username},</p>
    <p>You requested a password reset for your Bastion AI Workspace account.</p>
    <p><a href="{reset_url}">Click here to reset your password</a></p>
    <p>This link will expire in 1 hour.</p>
    <p>If you did not request this password reset, please ignore this email.</p>
    <hr>
    <p><small>Best regards,<br>Bastion AI Workspace</small></p>
</body>
</html>
"""
        
        return await self.send_email(to_email, subject, body_text, body_html)
    
    async def send_verification_email(
        self,
        to_email: str,
        username: str,
        verification_token: str,
        verification_url: str
    ) -> bool:
        """Send email verification email"""
        subject = "Verify Your Email - Bastion AI Workspace"
        
        body_text = f"""
Hello {username},

Please verify your email address by clicking the following link:
{verification_url}

If you did not create an account, please ignore this email.

Best regards,
Bastion AI Workspace
"""
        
        body_html = f"""
<html>
<body>
    <h2>Verify Your Email</h2>
    <p>Hello {username},</p>
    <p>Please verify your email address by clicking the following link:</p>
    <p><a href="{verification_url}">Verify Email Address</a></p>
    <p>If you did not create an account, please ignore this email.</p>
    <hr>
    <p><small>Best regards,<br>Bastion AI Workspace</small></p>
</body>
</html>
"""
        
        return await self.send_email(to_email, subject, body_text, body_html)


# Global email service instance
email_service = EmailService()

