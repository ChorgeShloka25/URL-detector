import re
import math
from collections import Counter
from urllib.parse import urlparse
import pandas as pd

# Regular expression pattern to check if hostname is an IPv4 address
IP_PATTERN = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')

# Common URL shortener services
SHORTENERS = [
    'bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 
    't.co', 'is.gd', 'cutt.ly', 'rb.gy'
]

# High-frequency suspicious keywords commonly used in phishing campaigns
SUSPICIOUS_KEYWORDS = [
    'login', 'signin', 'verify', 'update', 'banking', 
    'secure', 'account', 'confirm', 'password', 'suspend', 
    'alert', 'support', 'validate', 'unlock', 'billing', 'webscr'
]

# Targeted brand names frequently spoofed in subdomains or paths
TARGET_BRANDS = [
    'google', 'netflix', 'facebook', 'microsoft', 'apple', 
    'amazon', 'paypal', 'instagram', 'coinbase', 'dhl', 
    'paytm', 'chase', 'sbi', 'irctc', 'wellsfargo', 
    'fedex', 'binance', 'icici', 'hdfc', 'bankofamerica', 'whatsapp'
]

def sanitize_url(raw_url: str) -> str:
    """
    Ensure the URL has a scheme (defaults to http:// if missing) for correct parsing.
    """
    url = raw_url.strip()
    if not (url.startswith('http://') or url.startswith('https://')):
        url = 'http://' + url
    return url

def extract_features(raw_url: str) -> dict:
    """
    Extracts the 20 lexical, host, and behavioral features matching the dataset format.
    """
    # Ensure URL is parseable
    sanitized = sanitize_url(raw_url)
    p = urlparse(sanitized)
    
    # Extract host and check for userinfo (e.g. user@host)
    real_host = p.netloc.split('@')[-1].split(':')[0].strip().lower()
    is_ip = bool(IP_PATTERN.match(real_host))
    host_parts = real_host.split('.') if real_host else []
    
    # Feature 1: url_length
    url_length = len(raw_url.strip())
    
    # Feature 2: hostname_length
    hostname_length = len(real_host)
    
    # Feature 3: path_length
    path_length = len(p.path)
    
    # Feature 4: num_dots
    num_dots = raw_url.count('.')
    
    # Feature 5: num_hyphens
    num_hyphens = raw_url.count('-')
    
    # Feature 6: num_underscores
    num_underscores = raw_url.count('_')
    
    # Feature 7: num_digits
    num_digits = sum(c.isdigit() for c in raw_url)
    
    # Feature 8: num_special_chars (non-alphanumeric)
    num_special_chars = sum(not c.isalnum() for c in raw_url)
    
    # Feature 9: num_subdomains
    num_subdomains = max(0, len(host_parts) - 2) if not is_ip else 0
    
    # Feature 10: url_depth (number of non-empty path segments)
    path_segments = [seg for seg in p.path.split('/') if seg]
    url_depth = len(path_segments)
    
    # Feature 11: num_query_params
    num_query_params = len(p.query.split('&')) if p.query else 0
    
    # Feature 12: has_at_symbol
    has_at_symbol = 1 if '@' in raw_url else 0
    
    # Feature 13: has_https
    has_https = 1 if p.scheme.lower() == 'https' else 0
    
    # Feature 14: has_ip_address
    has_ip_address = 1 if is_ip else 0
    
    # Feature 15: has_shortener (checked on hostname + path)
    target_short = (real_host + p.path).lower()
    has_shortener = 1 if any(s in target_short for s in SHORTENERS) else 0
    
    # Feature 16: num_suspicious_keywords
    found_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in raw_url.lower()]
    num_suspicious_keywords = len(found_keywords)
    
    # Feature 17: has_brand_in_subdomain_or_path
    has_brand = 0
    matched_brand = None
    for b in TARGET_BRANDS:
        if b in raw_url.lower():
            if not (real_host == f'{b}.com' or real_host.endswith(f'.{b}.com')):
                has_brand = 1
                matched_brand = b
                break
            if b in p.path.lower() or ('@' in p.netloc and b in p.netloc.split('@')[0].lower()):
                has_brand = 1
                matched_brand = b
                break
                
    # Feature 18: digit_ratio
    digit_ratio = round(num_digits / url_length, 4) if url_length > 0 else 0.0
    
    # Feature 19: url_entropy (Shannon entropy)
    prob = [count / url_length for count in Counter(raw_url).values()] if url_length > 0 else [1]
    url_entropy = round(-sum(p_x * math.log2(p_x) for p_x in prob), 4)
    
    # Feature 20: tld
    tld = 'ip' if is_ip else (host_parts[-1] if host_parts else '')
    
    features = {
        'url_length': url_length,
        'hostname_length': hostname_length,
        'path_length': path_length,
        'num_dots': num_dots,
        'num_hyphens': num_hyphens,
        'num_underscores': num_underscores,
        'num_digits': num_digits,
        'num_special_chars': num_special_chars,
        'num_subdomains': num_subdomains,
        'url_depth': url_depth,
        'num_query_params': num_query_params,
        'has_at_symbol': has_at_symbol,
        'has_https': has_https,
        'has_ip_address': has_ip_address,
        'has_shortener': has_shortener,
        'num_suspicious_keywords': num_suspicious_keywords,
        'has_brand_in_subdomain_or_path': has_brand,
        'digit_ratio': digit_ratio,
        'url_entropy': url_entropy,
        'tld': tld
    }
    
    # Provide human-interpretable contextual metadata for UI explanations
    metadata = {
        'sanitized_url': sanitized,
        'hostname': real_host,
        'tld': tld,
        'found_keywords': found_keywords,
        'matched_brand': matched_brand,
        'is_ip': is_ip,
        'has_shortener': bool(has_shortener),
        'has_https': bool(has_https),
        'has_at_symbol': bool(has_at_symbol)
    }
    
    return features, metadata

def features_to_df(features: dict) -> pd.DataFrame:
    """
    Converts features dict to a single-row DataFrame suitable for model inference.
    """
    return pd.DataFrame([features])
