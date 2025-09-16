# security_enhancements.py
import time
import smtplib
import requests
from email.mime.text import MIMEText
from config.settings import settings
from config.database import execute_query

# ------------------------------
# Brute-force login tracking
# ------------------------------
login_attempts_cache = {}  # {username: [timestamp1, timestamp2, ...]}

def record_login_attempt(username, success):
    now = time.time()
    attempts = login_attempts_cache.get(username, [])
    # Remove old attempts beyond lockout duration
    attempts = [ts for ts in attempts if now - ts < settings.LOCKOUT_DURATION_MINUTES * 60]
    if not success:
        attempts.append(now)
    login_attempts_cache[username] = attempts

    # If user exceeded attempts, send email alert
    if len(attempts) >= settings.MAX_LOGIN_ATTEMPTS and not success:
        send_email_alert(
            to_email=settings.ADMIN_EMAIL,
            subject=f"Brute-force alert for {username}",
            body=f"User {username} exceeded max login attempts ({settings.MAX_LOGIN_ATTEMPTS})."
        )

def check_login_rate(username):
    attempts = login_attempts_cache.get(username, [])
    now = time.time()
    # Keep only recent attempts
    attempts = [ts for ts in attempts if now - ts < settings.LOCKOUT_DURATION_MINUTES * 60]
    login_attempts_cache[username] = attempts
    return len(attempts) < settings.MAX_LOGIN_ATTEMPTS

# ------------------------------
# Email alerts
# ------------------------------
def send_email_alert(to_email, subject, body):
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = settings.SMTP_USER
        msg['To'] = to_email

        server = smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT)
        server.starttls()
        server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
        server.sendmail(settings.SMTP_USER, [to_email], msg.as_string())
        server.quit()
    except Exception as e:
        print("Error sending email:", e)

# ------------------------------
# IP Geolocation alerts
# ------------------------------
geoip_cache = {}  # {user_id: last_country}

def get_country_from_ip(ip_address):
    try:
        url = f"https://api.ipgeolocation.io/ipgeo?apiKey={settings.GEOIP_API_KEY}&ip={ip_address}"
        response = requests.get(url)
        data = response.json()
        return data.get("country_name")
    except:
        return None

def check_geoip_alert(user_id, ip_address):
    country = get_country_from_ip(ip_address)
    if not country:
        return

    last_country = geoip_cache.get(user_id)
    if last_country and last_country != country:
        # Send admin alert
        send_email_alert(
            to_email=settings.ADMIN_EMAIL,
            subject=f"IP Geo Alert for User {user_id}",
            body=f"User logged in from new country: {country} (previous: {last_country})"
        )
    geoip_cache[user_id] = country
