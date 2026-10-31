import re
import socket
import ipaddress
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# ==============================================================================
# PHISHGUARD AI - BACKEND URL FEATURES & THREAT DETECTION ENGINE
# Designed for Machine Learning-Based Phishing Detection
# ==============================================================================

# Exact 30 feature names expected by the trained Random Forest model (feature_names.pkl)
FEATURE_NAMES = [
    "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
    "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State",
    "Domain_registeration_length", "Favicon", "port", "HTTPS_token", "Request_URL",
    "URL_of_Anchor", "Links_in_tags", "SFH", "Submitting_to_email", "Abnormal_URL",
    "Redirect", "on_mouseover", "RightClick", "popUpWidnow", "Iframe", "age_of_domain",
    "DNSRecord", "web_traffic", "Page_Rank", "Google_Index", "Links_pointing_to_page",
    "Statistical_report"
]

# Known URL shorteners that conceal destination addresses
SHORTENING_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "ow.ly", "buff.ly",
    "rebrand.ly", "cutt.ly", "bit.do", "t.ly", "shorturl.at", "tiny.cc",
    "bc.vc", "bl.ink", "v.gd"
}

# Major protected brands frequently targeted in online phishing campaigns
PROTECTED_BRANDS = {
    "google": [
        "google.com", "google.co.in", "google.co.uk", "google.ca",
        "google.de", "google.fr", "google.com.au", "google.com.br",
        "google.co.jp", "google.com.co"
    ],
    "paypal": ["paypal.com", "paypal.me"],
    "microsoft": ["microsoft.com", "live.com", "office.com", "outlook.com", "msn.com"],
    "apple": ["apple.com", "icloud.com"],
    "amazon": ["amazon.com", "amazon.co.uk", "amazon.in", "amazon.de", "amazon.ca", "amazon.co.jp"],
    "netflix": ["netflix.com"],
    "facebook": ["facebook.com", "fb.com", "meta.com"],
    "instagram": ["instagram.com"],
    "whatsapp": ["whatsapp.com"],
    "chase": ["chase.com"],
    "bankofamerica": ["bankofamerica.com"],
    "wellsfargo": ["wellsfargo.com"],
    "github": ["github.com"],
    "dropbox": ["dropbox.com"],
    "linkedin": ["linkedin.com"],
    "twitter": ["twitter.com", "x.com"],
}

# Homoglyph character substitutions commonly used to deceive users (e.g., paypa1, g00gle)
HOMOGLYPH_MAP = {
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "@": "a", "vv": "w", "rn": "m"
}

# Sensitive credential and financial action keywords
SENSITIVE_KEYWORDS = {
    "login", "signin", "verify", "verification", "secure", "security",
    "account", "update", "banking", "wallet", "support", "service",
    "password", "auth", "confirm", "validation", "recover"
}

# Common multi-part country code top-level domains (ccTLDs)
MULTI_PART_TLDS = {
    "co.uk", "co.in", "com.au", "com.br", "co.jp", "co.nz", "co.za",
    "gov.in", "edu.au", "org.uk", "com.mx", "com.sg", "com.tr", "com.co"
}

# Top-level domains historically associated with high abuse in spam and phishing campaigns
HIGH_RISK_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "top", "buzz", "work", "click", "rest", "stream"
}


# ==============================================================================
# INPUT VALIDATION HELPERS
# ==============================================================================

def is_email_address(text):
    """
    Check whether the input is an email address rather than a website link.
    Rejects email inputs before feature extraction or model prediction.
    """
    text = (text or "").strip()
    if text.lower().startswith("mailto:"):
        text = text[7:]
    email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(email_pattern, text))


def is_ip_address(hostname):
    """Check whether the hostname is a numerical IP address instead of a domain name."""
    try:
        ipaddress.ip_address(hostname)
        return True
    except (ValueError, AttributeError):
        return False


def is_shortening_service(hostname):
    """Check whether the hostname belongs to a known URL shortening service."""
    hostname = (hostname or "").lower().rstrip(".")
    return any(
        hostname == domain or hostname.endswith("." + domain)
        for domain in SHORTENING_DOMAINS
    )


def normalize_homoglyphs(text):
    """
    Convert visually deceptive characters into standard Latin letters.
    For example: 'paypa1' becomes 'paypal', 'g00gle' becomes 'google'.
    """
    text = (text or "").lower()
    for char, replacement in HOMOGLYPH_MAP.items():
        text = text.replace(char, replacement)
    return text


def split_domain_parts(hostname):
    """
    Split a hostname into subdomains, main domain, and effective TLD.
    Properly handles multi-part TLDs (e.g., .co.uk, .com.au, .co.in).
    """
    hostname = (hostname or "").lower().rstrip(".")
    parts = hostname.split(".")

    if len(parts) < 2:
        return [], hostname, "", hostname

    if len(parts) >= 3 and f"{parts[-2]}.{parts[-1]}" in MULTI_PART_TLDS:
        effective_tld = f"{parts[-2]}.{parts[-1]}"
        main_domain = parts[-3]
        subdomains = parts[:-3]
    else:
        effective_tld = parts[-1]
        main_domain = parts[-2]
        subdomains = parts[:-2]

    root_domain = f"{main_domain}.{effective_tld}"
    return subdomains, main_domain, effective_tld, root_domain


def validate_url_input(raw_input):
    """
    Validate and normalize user input before prediction.
    Returns:
        (is_valid: bool, category: str, message: str, normalized_url: str)
    """
    raw_input = (raw_input or "").strip()

    if not raw_input:
        return False, "empty", "Please type or paste a website address first.", ""

    if is_email_address(raw_input):
        return (
            False,
            "email",
            "It looks like you entered an email address. PhishGuard AI is designed to "
            "scan website links (like example.com), not email addresses. Please enter a website link to check.",
            ""
        )

    if any(character.isspace() for character in raw_input):
        return False, "spaces", "Please make sure the website address does not contain any spaces.", ""

    # Check if a protocol/scheme was provided
    scheme_match = re.match(r"^([a-zA-Z][a-zA-Z0-9+.-]*):?//", raw_input)
    if scheme_match:
        scheme = scheme_match.group(1).lower()
        if scheme not in ("http", "https"):
            return (
                False,
                "protocol",
                "Please enter a web address starting with http://, https://, or a domain like example.com.",
                ""
            )
        candidate = raw_input
    else:
        candidate = "https://" + raw_input

    try:
        parsed = urlparse(candidate)
        hostname = (parsed.hostname or "").lower()

        if not hostname:
            return False, "invalid", "Please enter a valid website link, like example.com or https://www.example.com", ""

        if parsed.scheme.lower() not in ("http", "https"):
            return False, "protocol", "Please enter a web address starting with http://, https://, or a domain like example.com.", ""

        if not is_ip_address(hostname):
            if "." not in hostname or any(len(part) == 0 for part in hostname.split(".")):
                return False, "invalid", "Please enter a complete website address with a domain ending, like example.com", ""

        return True, "valid", "", candidate

    except Exception:
        return False, "invalid", "Please enter a valid website link, like example.com or https://www.example.com", ""


# ==============================================================================
# SAFE TIMEOUT-ENABLED DNS RESOLVER
# ==============================================================================

def check_domain_dns(hostname, timeout=3.0):
    """
    Check whether a hostname exists and resolves online via standard DNS lookup.
    Uses a safe timeout to avoid blocking the application if a domain hangs.

    Returns:
        (is_resolved: bool, status_code: str)
        where status_code is 'resolved', 'unresolved', or 'timeout'.
    """
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(socket.getaddrinfo, hostname, None)
            future.result(timeout=timeout)
            return True, "resolved"
    except TimeoutError:
        return False, "timeout"
    except (socket.gaierror, OSError):
        return False, "unresolved"
    except Exception as e:
        return False, f"error: {e}"


# ==============================================================================
# LOOKALIKE AND SUSPICIOUS DOMAIN DETECTION
# ==============================================================================

def detect_lookalike_and_threats(hostname, parsed_path=""):
    """
    Analyzes domain structure for lookalike and impersonation techniques:
    1. Brand typosquatting / homoglyphs (e.g., paypa1.com, g00gle.com)
    2. Brand name combined with sensitive security keywords (e.g., paypal-security.com)
    3. Brand name disguised in a subdomain of an unrelated domain (e.g., apple.com.attacker.com)
    4. Unofficial brand lookalike domains (e.g., google.co vs legitimate google.com)
    5. Combinations of suspicious indicators (high-risk TLD + sensitive keywords)

    Note: Normal dictionary words (like applepie.com or apply.com) and modern TLDs
    (like .xyz or .tech) are NOT flagged on their own, preventing false positives.

    Returns:
        is_suspicious (bool): True if high-confidence phishing indicators are detected
        reasons (list of str): Plain-language explanations of any detected threats
        warnings (list of str): Informational notes that do not trigger a phishing verdict
    """
    subdomains, main_domain, effective_tld, root_domain = split_domain_parts(hostname)

    reasons = []
    warnings = []

    # 0. Check for raw IP address in URL
    if is_ip_address(hostname):
        reasons.append(
            "This address uses a raw number instead of a regular website name, which is often done to hide who owns the site."
        )

    if not main_domain and not is_ip_address(hostname):
        return False, reasons, warnings

    normalized_main = normalize_homoglyphs(main_domain)
    # Split domain tokens by hyphens to avoid false positives on words like 'applepie' or 'pineapple'
    tokens = re.split(r"[-_]", main_domain)

    # 1. Brand Impersonation and Lookalike Detection
    for brand, legit_domains in PROTECTED_BRANDS.items():

        # Case A: Exact brand name on an unauthorized TLD (e.g., google.co vs google.com)
        if main_domain == brand and root_domain not in legit_domains:
            reasons.append(
                f"This address looks like '{brand.title()}', but it isn't their official website ({legit_domains[0]})."
            )

        # Case B: Visually deceptive letters/numbers (e.g., paypa1.com, g00gle.com)
        elif main_domain != brand and normalized_main == brand:
            reasons.append(
                f"This address uses lookalike letters or numbers to imitate '{brand.title()}' (like swapping letters with similar-looking numbers)."
            )

        # Case C: Brand token paired with a sensitive credential keyword (e.g. paypal-security.com)
        elif brand in tokens and root_domain not in legit_domains:
            for kw in SENSITIVE_KEYWORDS:
                if kw in tokens:
                    reasons.append(
                        f"This address combines the name '{brand.title()}' with a sensitive word ('{kw}') on an unofficial website."
                    )
                    break

        # Case D: Trusted brand name disguised inside subdomains (e.g., apple.com.fake-login.info)
        for sub in subdomains:
            if (sub == brand or f"{brand}.com" in sub) and root_domain not in legit_domains:
                reasons.append(
                    f"This address puts '{brand.title()}' in the subdomain to look genuine, but the actual website is hosted on '{root_domain}'."
                )
                break

    # 2. High-Risk TLD Evaluation (Avoids False Positives)
    # A suspicious TLD alone is NOT treated as phishing; it requires corroborating evidence.
    if effective_tld in HIGH_RISK_TLDS:
        has_sensitive_kw = any(kw in main_domain for kw in SENSITIVE_KEYWORDS)
        has_excessive_hyphens = main_domain.count("-") >= 2

        if has_sensitive_kw or has_excessive_hyphens:
            reasons.append(
                f"This website pairs a high-risk web extension (.{effective_tld}) with sensitive words or multiple hyphens."
            )
        else:
            warnings.append(
                f"Note: This website uses a .{effective_tld} web extension, which is often used for short-term or low-cost sites."
            )

    is_suspicious = len(reasons) > 0
    return is_suspicious, reasons, warnings


# ==============================================================================
# FEATURE EXTRACTION FOR MACHINE LEARNING MODEL
# ==============================================================================

def extract_url_features(url, dns_resolved=True, scheme_specified=True):
    """
    Extract 30 features matching the exact format and order expected by the
    trained Random Forest model (feature_names.pkl).

    Dataset convention:
    -1 = suspicious characteristic
     0 = uncertain or unavailable
     1 = legitimate-looking characteristic

    Parameters:
        url (str): The candidate website URL to analyze.
        dns_resolved (bool): Result of DNS address lookup (True if domain resolves).
        scheme_specified (bool): True if the user explicitly typed http:// or https://.
    """
    url = url.strip()

    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url

    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    full_url = url.lower()

    subdomains, main_domain, effective_tld, root_domain = split_domain_parts(hostname)

    features = {}

    # 1. IP address used instead of hostname
    features["having_IP_Address"] = -1 if is_ip_address(hostname) else 1

    # 2. URL Length
    if len(url) < 54:
        features["URL_Length"] = 1
    elif len(url) <= 75:
        features["URL_Length"] = 0
    else:
        features["URL_Length"] = -1

    # 3. URL Shortening Service
    features["Shortining_Service"] = -1 if is_shortening_service(hostname) else 1

    # 4. @ Symbol in URL (causes browsers to ignore preceding text)
    features["having_At_Symbol"] = -1 if "@" in url else 1

    # 5. Double slash redirection after the scheme
    remainder = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
    features["double_slash_redirecting"] = -1 if "//" in remainder else 1

    # 6. Prefix or Suffix hyphen in hostname
    features["Prefix_Suffix"] = -1 if "-" in hostname else 1

    # 7. Subdomain count (accounting for multi-part TLDs)
    if len(subdomains) == 0:
        features["having_Sub_Domain"] = 1
    elif len(subdomains) == 1:
        features["having_Sub_Domain"] = 0
    else:
        features["having_Sub_Domain"] = -1

    # 8. SSL / HTTPS Presence
    # If the user explicitly typed https://, set to 1.
    # If the user explicitly typed http://, set to -1.
    # If the scheme was auto-prepended, set to 0 (uncertain) to avoid artificial bias.
    if scheme_specified:
        features["SSLfinal_State"] = 1 if parsed.scheme.lower() == "https" else -1
    else:
        features["SSLfinal_State"] = 0

    # 9. Domain registration length (requires live paid WHOIS API)
    features["Domain_registeration_length"] = 0

    # 10. Favicon (requires fetching webpage HTML)
    features["Favicon"] = 0

    # 11. Port validation
    try:
        port = parsed.port
        if port is None or port in (80, 443):
            features["port"] = 1
        else:
            features["port"] = -1
    except ValueError:
        features["port"] = 0

    # 12. HTTPS token inside domain name (e.g. http://https-login.com)
    features["HTTPS_token"] = -1 if "https" in hostname else 1

    # 13-16. Webpage content features (require webpage HTML inspection)
    features["Request_URL"] = 0
    features["URL_of_Anchor"] = 0
    features["Links_in_tags"] = 0
    features["SFH"] = 0

    # 17. Submitting information to email (e.g. mailto: in URL)
    features["Submitting_to_email"] = -1 if ("mailto:" in full_url or "mail(" in full_url) else 1

    # 18. Abnormal URL (checks for credential-harvesting path keywords on unauthorized domains)
    sensitive_path_pattern = r"/(login|signin|account|verify|update|auth|password)/(google|paypal|apple|microsoft|chase)"
    path_has_brand_auth = bool(re.search(sensitive_path_pattern, parsed.path.lower())) and (
        root_domain not in [d for doms in PROTECTED_BRANDS.values() for d in doms]
    )
    features["Abnormal_URL"] = -1 if path_has_brand_auth else 1

    # 19. Redirect parameter in query string
    has_redirect_param = bool(
        re.search(r"[?&](redirect|url|next|dest|target)=https?://", full_url)
    )
    features["Redirect"] = 0 if has_redirect_param else 1

    # 20-24. Webpage interaction features (require HTML / JS inspection)
    features["on_mouseover"] = 0
    features["RightClick"] = 0
    features["popUpWidnow"] = 0
    features["Iframe"] = 0
    features["age_of_domain"] = 0

    # 25. DNS Record (Passed directly from DNS validation check)
    features["DNSRecord"] = 1 if dns_resolved else -1

    # 26-30. External reputation and traffic features
    features["web_traffic"] = 0
    features["Page_Rank"] = 0
    features["Google_Index"] = 0
    features["Links_pointing_to_page"] = 0
    features["Statistical_report"] = 0

    # Ensure all 30 features are returned in the exact expected order
    return {name: features[name] for name in FEATURE_NAMES}
