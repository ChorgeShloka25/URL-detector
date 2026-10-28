import re
import math
from collections import Counter
from urllib.parse import urlparse
import pandas as pd

# --- Constants ---

# List of known URL shortener domains
SHORTENERS = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 'cutt.ly', 'rb.gy']

# Words often used in phishing URLs
SUSPICIOUS_KEYWORDS = [
    'login', 'signin', 'verify', 'update', 'banking', 'secure',
    'account', 'confirm', 'password', 'suspend', 'alert', 'support',
    'validate', 'unlock', 'billing', 'webscr'
]

# Brand names that phishers commonly impersonate
TARGET_BRANDS = [
    'google', 'netflix', 'facebook', 'microsoft', 'apple', 'amazon',
    'paypal', 'instagram', 'coinbase', 'dhl', 'paytm', 'chase',
    'sbi', 'irctc', 'wellsfargo', 'fedex', 'binance', 'icici',
    'hdfc', 'bankofamerica', 'whatsapp'
]

# Regex to detect if a host is a raw IP (e.g. 192.168.1.1)
IP_PATTERN = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')


def extract_features(raw_url):
    """
    Takes a URL string and returns 20 features used by the ML model.
    Also returns metadata to help explain the result in the UI.
    """

    # Step 1: Add http:// if scheme is missing so urlparse works correctly
    url = raw_url.strip()
    if not (url.startswith('http://') or url.startswith('https://')):
        url = 'http://' + url

    parsed = urlparse(url)

    # Step 2: Extract the clean hostname (strip user@host and port)
    host = parsed.netloc.split('@')[-1].split(':')[0].strip().lower()
    is_ip = bool(IP_PATTERN.match(host))
    host_parts = host.split('.') if host else []

    # Step 3: Count suspicious keywords found in the URL
    found_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in url.lower()]

    # Step 4: Check if a brand name appears somewhere suspicious
    has_brand, matched_brand = 0, None
    for brand in TARGET_BRANDS:
        if brand in url.lower():
            # Not suspicious if the URL actually belongs to that brand
            if not (host == f'{brand}.com' or host.endswith(f'.{brand}.com')):
                has_brand, matched_brand = 1, brand
                break
            if brand in parsed.path.lower():
                has_brand, matched_brand = 1, brand
                break

    # Step 5: Calculate Shannon entropy (how random the URL characters are)
    url_len = len(raw_url.strip())
    if url_len > 0:
        counts = Counter(raw_url)
        entropy = round(-sum((c / url_len) * math.log2(c / url_len) for c in counts.values()), 4)
    else:
        entropy = 0.0

    # Step 6: Get the top-level domain (.com, .xyz, etc.)
    tld = 'ip' if is_ip else (host_parts[-1] if host_parts else '')

    # --- Build feature dictionary (must match training data column names) ---
    num_digits = sum(c.isdigit() for c in raw_url)

    features = {
        'url_length':                     url_len,
        'hostname_length':                len(host),
        'path_length':                    len(parsed.path),
        'num_dots':                       raw_url.count('.'),
        'num_hyphens':                    raw_url.count('-'),
        'num_underscores':                raw_url.count('_'),
        'num_digits':                     num_digits,
        'num_special_chars':              sum(not c.isalnum() for c in raw_url),
        'num_subdomains':                 max(0, len(host_parts) - 2) if not is_ip else 0,
        'url_depth':                      len([s for s in parsed.path.split('/') if s]),
        'num_query_params':               len(parsed.query.split('&')) if parsed.query else 0,
        'has_at_symbol':                  1 if '@' in raw_url else 0,
        'has_https':                      1 if parsed.scheme == 'https' else 0,
        'has_ip_address':                 1 if is_ip else 0,
        'has_shortener':                  1 if any(s in host for s in SHORTENERS) else 0,
        'num_suspicious_keywords':        len(found_keywords),
        'has_brand_in_subdomain_or_path': has_brand,
        'digit_ratio':                    round(num_digits / url_len, 4) if url_len > 0 else 0.0,
        'url_entropy':                    entropy,
        'tld':                            tld,
    }

    meta = {
        'found_keywords': found_keywords,
        'matched_brand': matched_brand,
        'hostname': host,
    }   
    
    return features, meta


def features_to_df(features):
    """Convert a feature dict into a single-row DataFrame for model prediction."""
    return pd.DataFrame([features])
