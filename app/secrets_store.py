import os

try:
    import keyring
except ImportError:  # pragma: no cover
    keyring = None

SERVICE_NAME = "ai_sourcing_mvp"
KEYS = {
    "OPENAI_API_KEY": "openai_api_key",
    "SERPAPI_API_KEY": "serpapi_api_key",
    "GCP_OCR_BUCKET": "gcp_ocr_bucket",
    "GOOGLE_APPLICATION_CREDENTIALS": "gcp_credentials_path",
    "EXA_API_KEY": "exa_api_key",
    "PROXYCURL_API_KEY": "proxycurl_api_key",
}


def keyring_available() -> bool:
    return keyring is not None


def get_key(env_name: str) -> str | None:
    if not keyring:
        return None
    key_name = KEYS.get(env_name)
    if not key_name:
        return None
    try:
        return keyring.get_password(SERVICE_NAME, key_name)
    except Exception:
        return None


def set_key(env_name: str, value: str) -> bool:
    if not keyring:
        return False
    key_name = KEYS.get(env_name)
    if not key_name:
        return False
    try:
        keyring.set_password(SERVICE_NAME, key_name, value)
        return True
    except Exception:
        return False


def delete_key(env_name: str) -> bool:
    if not keyring:
        return False
    key_name = KEYS.get(env_name)
    if not key_name:
        return False
    try:
        keyring.delete_password(SERVICE_NAME, key_name)
        return True
    except Exception:
        return False


def ensure_key(env_name: str) -> bool:
    if os.getenv(env_name):
        return True
    key = get_key(env_name)
    if key:
        os.environ[env_name] = key
        return True
    return False


def ensure_openai_key() -> bool:
    return ensure_key("OPENAI_API_KEY")


def ensure_serpapi_key() -> bool:
    return ensure_key("SERPAPI_API_KEY")


def ensure_gcp_bucket() -> bool:
    return ensure_key("GCP_OCR_BUCKET")


def ensure_gcp_credentials() -> bool:
    return ensure_key("GOOGLE_APPLICATION_CREDENTIALS")
