import sys
import unittest
from unittest.mock import patch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import joblib
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# Import backend modules
from url_features import (
    FEATURE_NAMES,
    validate_url_input,
    is_email_address,
    is_ip_address,
    is_shortening_service,
    split_domain_parts,
    detect_lookalike_and_threats,
    extract_url_features,
    check_domain_dns
)


class TestInputValidation(unittest.TestCase):
    """Test URL and input format validation, including email address filtering."""

    def test_email_address_rejected(self):
        valid, category, msg, _ = validate_url_input("pnombal@gmail.com")
        self.assertFalse(valid)
        self.assertEqual(category, "email")
        self.assertIn("scan website links", msg)

    def test_mailto_email_rejected(self):
        valid, category, msg, _ = validate_url_input("mailto:support@company.org")
        self.assertFalse(valid)
        self.assertEqual(category, "email")

    def test_empty_input_rejected(self):
        valid, category, msg, _ = validate_url_input("   ")
        self.assertFalse(valid)
        self.assertEqual(category, "empty")

    def test_spaces_in_url_rejected(self):
        valid, category, msg, _ = validate_url_input("https://google.com/test page")
        self.assertFalse(valid)
        self.assertEqual(category, "spaces")

    def test_unsupported_protocol_rejected(self):
        valid, category, msg, _ = validate_url_input("ftp://ftp.example.com")
        self.assertFalse(valid)
        self.assertEqual(category, "protocol")

    def test_invalid_domain_format_rejected(self):
        valid, category, msg, _ = validate_url_input("notadomain")
        self.assertFalse(valid)
        self.assertEqual(category, "invalid")

    def test_valid_url_without_scheme(self):
        valid, category, msg, candidate = validate_url_input("google.com")
        self.assertTrue(valid)
        self.assertEqual(candidate, "https://google.com")

    def test_valid_url_with_https(self):
        valid, category, msg, candidate = validate_url_input("https://www.google.com")
        self.assertTrue(valid)
        self.assertEqual(candidate, "https://www.google.com")

    def test_valid_url_with_http(self):
        valid, category, msg, candidate = validate_url_input("http://example.com/login")
        self.assertTrue(valid)
        self.assertEqual(candidate, "http://example.com/login")


class TestFeatureExtraction(unittest.TestCase):
    """Test the integrity, feature count, and ordering of extracted URL features."""

    def setUp(self):
        self.model = joblib.load(PROJECT_ROOT / "phishing_model.pkl")
        self.feature_names = joblib.load(PROJECT_ROOT / "feature_names.pkl")

    def test_all_30_features_present_and_ordered(self):
        features = extract_url_features("https://www.google.com")
        self.assertEqual(len(features), 30)
        self.assertEqual(list(features.keys()), self.feature_names)

    def test_feature_values_in_valid_range(self):
        features = extract_url_features("https://github.com")
        for name, value in features.items():
            self.assertIn(value, (-1, 0, 1), f"Feature {name} has unexpected value {value}")

    def test_ip_address_feature(self):
        features = extract_url_features("http://192.168.1.1/admin")
        self.assertEqual(features["having_IP_Address"], -1)

    def test_shortening_service_feature(self):
        features = extract_url_features("https://bit.ly/secure-login")
        self.assertEqual(features["Shortining_Service"], -1)

    def test_at_symbol_feature(self):
        features = extract_url_features("http://google.com@phishing.site")
        self.assertEqual(features["having_At_Symbol"], -1)

    def test_double_slash_redirecting(self):
        features = extract_url_features("http://example.com//phish")
        self.assertEqual(features["double_slash_redirecting"], -1)

    def test_subdomain_handling_with_multi_part_tld(self):
        # google.co.uk has NO subdomains
        features = extract_url_features("https://google.co.uk")
        self.assertEqual(features["having_Sub_Domain"], 1)

        # www.google.co.uk has 1 subdomain (www)
        features_www = extract_url_features("https://www.google.co.uk")
        self.assertEqual(features_www["having_Sub_Domain"], 0)

        # login.verify.google.co.uk has 2 subdomains
        features_multi = extract_url_features("https://login.verify.google.co.uk")
        self.assertEqual(features_multi["having_Sub_Domain"], -1)

    def test_ssl_scheme_specified_vs_unspecified(self):
        # When explicitly typed https://
        feat_explicit_https = extract_url_features("https://example.com", scheme_specified=True)
        self.assertEqual(feat_explicit_https["SSLfinal_State"], 1)

        # When explicitly typed http://
        feat_explicit_http = extract_url_features("http://example.com", scheme_specified=True)
        self.assertEqual(feat_explicit_http["SSLfinal_State"], -1)

        # When auto-prepended
        feat_auto = extract_url_features("example.com", scheme_specified=False)
        self.assertEqual(feat_auto["SSLfinal_State"], 0)


class TestLookalikeAndBrandSecurityRules(unittest.TestCase):
    """Test deterministic lookalike and brand impersonation detection rules."""

    def test_homoglyph_paypa1(self):
        susp, reasons, _ = detect_lookalike_and_threats("paypa1.com")
        self.assertTrue(susp)
        self.assertTrue(any("Paypal" in r for r in reasons))

    def test_homoglyph_g00gle(self):
        susp, reasons, _ = detect_lookalike_and_threats("g00gle.com")
        self.assertTrue(susp)
        self.assertTrue(any("Google" in r for r in reasons))

    def test_lookalike_google_co(self):
        susp, reasons, _ = detect_lookalike_and_threats("www.google.co")
        self.assertTrue(susp)
        self.assertTrue(any("Google" in r for r in reasons))

    def test_brand_with_sensitive_keyword(self):
        susp, reasons, _ = detect_lookalike_and_threats("paypal-security.com")
        self.assertTrue(susp)
        self.assertTrue(any("Paypal" in r and "security" in r for r in reasons))

    def test_subdomain_brand_disguise(self):
        susp, reasons, _ = detect_lookalike_and_threats("appleid.apple.com.fake-login.info")
        self.assertTrue(susp)
        self.assertTrue(any("Apple" in r and "subdomain" in r for r in reasons))

    def test_raw_ip_address_host(self):
        susp, reasons, _ = detect_lookalike_and_threats("192.168.1.1")
        self.assertTrue(susp)
        self.assertTrue(any("raw number" in r or "IP" in r for r in reasons))

    def test_false_positive_prevention_normal_words(self):
        # Normal dictionary words containing brand substrings must NOT be flagged
        for safe_domain in ["applepie.com", "pineapple.com", "apply.com"]:
            susp, reasons, _ = detect_lookalike_and_threats(safe_domain)
            self.assertFalse(susp, f"Domain {safe_domain} was falsely flagged: {reasons}")

    def test_false_positive_prevention_modern_tlds(self):
        # A .xyz or .tech domain without phishing keywords must not be flagged as phishing
        susp, reasons, warnings = detect_lookalike_and_threats("myportfolio.xyz")
        self.assertFalse(susp)

    def test_legitimate_brands_not_flagged(self):
        legit_domains = [
            "google.com", "www.google.com", "google.co.in", "google.co.uk",
            "paypal.com", "apple.com", "github.com", "amazon.co.uk", "microsoft.com"
        ]
        for d in legit_domains:
            susp, reasons, _ = detect_lookalike_and_threats(d)
            self.assertFalse(susp, f"Legitimate domain {d} was falsely flagged: {reasons}")


class TestDeterministicOfflinePipeline(unittest.TestCase):
    """Test the complete hybrid workflow with mocked DNS to ensure zero internet dependence."""

    def setUp(self):
        self.model = joblib.load(PROJECT_ROOT / "phishing_model.pkl")
        self.feature_names = joblib.load(PROJECT_ROOT / "feature_names.pkl")

    def simulate_pipeline(self, raw_input, mock_dns_status):
        valid, kind, msg, candidate = validate_url_input(raw_input)
        if not valid:
            return kind, msg

        from urllib.parse import urlparse
        parsed = urlparse(candidate)
        hostname = (parsed.hostname or "").lower()

        is_lookalike_suspicious, lookalike_reasons, _ = detect_lookalike_and_threats(hostname)
        if is_lookalike_suspicious:
            return "suspicious_lookalike", lookalike_reasons[0]

        if mock_dns_status != "resolved":
            return "unverified", mock_dns_status

        scheme_specified = bool(raw_input.startswith("http://") or raw_input.startswith("https://"))
        features = extract_url_features(candidate, dns_resolved=True, scheme_specified=scheme_specified)
        df = pd.DataFrame([[features[name] for name in self.feature_names]], columns=self.feature_names)
        pred = int(self.model.predict(df)[0])

        if pred == 1:
            return "suspicious_ml", "Machine learning flagged structural anomalies"
        else:
            return "likely_safe", "Normal patterns"

    def test_legitimate_site_offline(self):
        outcome, _ = self.simulate_pipeline("https://www.google.com", mock_dns_status="resolved")
        self.assertEqual(outcome, "likely_safe")

    def test_lookalike_offline(self):
        outcome, _ = self.simulate_pipeline("https://paypa1.com", mock_dns_status="resolved")
        self.assertEqual(outcome, "suspicious_lookalike")

    def test_unresolved_domain_offline(self):
        outcome, _ = self.simulate_pipeline("https://notexistingdomain999.com", mock_dns_status="unresolved")
        self.assertEqual(outcome, "unverified")

    def test_dns_timeout_offline(self):
        outcome, _ = self.simulate_pipeline("https://slowdomain.com", mock_dns_status="timeout")
        self.assertEqual(outcome, "unverified")

    def test_email_input_offline(self):
        outcome, _ = self.simulate_pipeline("pnombal@gmail.com", mock_dns_status="resolved")
        self.assertEqual(outcome, "email")


class TestLabeledModelBenchmark(unittest.TestCase):
    """
    Evaluate the Random Forest model on the labeled UCI Phishing dataset.
    Measures precision, recall, F1, FPR, and FNR deterministically on a test split.
    """

    def test_benchmark_metrics(self):
        dataset_path = PROJECT_ROOT / "dataset" / "Training Dataset.arff"
        self.assertTrue(dataset_path.exists())

        data, metadata = arff.loadarff(dataset_path)
        df = pd.DataFrame(data)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])

        X = df.drop(columns=["Result"])
        y = df["Result"].map({-1: 1, 1: 0})  # 1 = Phishing, 0 = Legitimate

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        model = joblib.load(PROJECT_ROOT / "phishing_model.pkl")
        feature_names = joblib.load(PROJECT_ROOT / "feature_names.pkl")

        X_test = X_test[feature_names]
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
        tp, fn = cm[0]
        fp, tn = cm[1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1)
        rec = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)

        # Baseline performance sanity thresholds
        self.assertGreater(acc, 0.95, f"Accuracy {acc:.4f} below 95%")
        self.assertGreater(prec, 0.95, f"Precision {prec:.4f} below 95%")
        self.assertGreater(rec, 0.95, f"Recall {rec:.4f} below 95%")
        self.assertGreater(f1, 0.95, f"F1-score {f1:.4f} below 95%")
        self.assertLess(fpr, 0.05, f"FPR {fpr:.4f} above 5%")
        self.assertLess(fnr, 0.05, f"FNR {fnr:.4f} above 5%")


if __name__ == "__main__":
    unittest.main()
