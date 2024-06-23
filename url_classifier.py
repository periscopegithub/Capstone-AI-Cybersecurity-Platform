import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from urllib.parse import urlparse
import tldextract
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from flask import jsonify

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

sensitive_words = [
    "secure",
    "account",
    "webscr",
    "login",
    "ebayisapi",
    "signin",
    "banking",
    "confirm",
]


def extract_domain_names(text):
    domain_pattern = r"[\w-]+\.[\w.-]+"
    return re.findall(domain_pattern, text)


def compute_features(url, feature_list):
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname if parsed_url.hostname else ""
    path = parsed_url.path.lower() if parsed_url.path else ""
    query = parsed_url.query if parsed_url.query else ""
    ext = tldextract.extract(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

    extracted = tldextract.extract(url)
    subdomain = extracted.subdomain
    domain = extracted.domain
    suffix = extracted.suffix

    subdomain_parts = subdomain.split(".") if subdomain else []
    subdomain_level = len(subdomain_parts) if subdomain else 0

    domain_in_subdomains = (
        1
        if extracted.subdomain
        and extracted.domain
        and extracted.domain in extracted.subdomain
        else 0
    )
    domain_in_paths = 1 if extracted.domain and extracted.domain in path else 0

    try:
        response = requests.get(url, timeout=10)
        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")
    except requests.RequestException:
        html_content = ""
        soup = None

    num_sensitive_words = sum(word in url.lower() for word in sensitive_words)

    features = {
        "NumDots": url.count("."),
        "SubdomainLevel": subdomain_level,
        "PathLevel": path.count("/") - 1 if path else 0,
        "UrlLength": len(url),
        "NumDash": url.count("-"),
        "NumDashInHostname": hostname.count("-") if hostname else 0,
        "AtSymbol": 1 if "@" in url else 0,
        "TildeSymbol": 1 if "~" in url else 0,
        "NumUnderscore": url.count("_"),
        "NumPercent": url.count("%"),
        "NumQueryComponents": query.count("="),
        "NumAmpersand": url.count("&"),
        "NumHash": url.count("#"),
        "NumNumericChars": sum(c.isdigit() for c in url),
        "NoHttps": 0 if "https" in url[:8] else 1,
        "RandomString": 1 if re.search(r"[\W_]+", url) else 0,
        "IpAddress": 1 if re.match(r"^\d+\.\d+\.\d+\.\d+$", hostname) else 0,
        "DomainInSubdomains": domain_in_subdomains,
        "DomainInPaths": domain_in_paths,
        "HttpsInHostname": 1 if "https" in hostname else 0,
        "HostnameLength": len(hostname),
        "PathLength": len(path),
        "QueryLength": len(query),
        "DoubleSlashInPath": 1 if "//" in path[1:] else 0,
        "NumSensitiveWords": num_sensitive_words,
        "EmbeddedBrandName": 0,
        "PctExtHyperlinks": 0,
        "PctExtResourceUrls": 0,
        "ExtFavicon": 0,
        "InsecureForms": 0,
        "RelativeFormAction": 0,
        "ExtFormAction": 0,
        "AbnormalFormAction": 0,
        "PctNullSelfRedirectHyperlinks": 0,
        "FrequentDomainNameMismatch": 0,
        "FakeLinkInStatusBar": 0,
        "RightClickDisabled": 0,
        "PopUpWindow": 0,
        "SubmitInfoToEmail": 0,
        "IframeOrFrame": 0,
        "MissingTitle": 0,
        "ImagesOnlyInForm": 0,
        "SubdomainLevelRT": hostname.count("."),
        "UrlLengthRT": len(url),
        "PctExtResourceUrlsRT": 0,
        "AbnormalExtFormActionR": 0,
        "ExtMetaScriptLinkRT": 0,
        "PctExtNullSelfRedirectHyperlinksRT": 0,
    }

    if soup:
        links = soup.find_all("a", href=True)
        total_links = len(links)
        null_or_self_links = 0
        external_links = 0

        for link in links:
            href = link["href"]
            if href.startswith("#") or href == "" or href.startswith("javascript"):
                null_or_self_links += 1
            elif urlparse(href).netloc and urlparse(href).netloc != hostname:
                external_links += 1

        features["PctExtHyperlinks"] = (
            (external_links / total_links) * 100 if total_links else 0
        )
        features["PctNullSelfRedirectHyperlinks"] = (
            (null_or_self_links / total_links) * 100 if total_links else 0
        )

        resources = soup.find_all(["script", "link", "img"], src=True)
        total_resources = len(resources)
        external_resources = sum(
            1
            for res in resources
            if urlparse(res["src"]).netloc and urlparse(res["src"]).netloc != hostname
        )
        features["PctExtResourceUrls"] = (
            (external_resources / total_resources) * 100 if total_resources > 0 else 0
        )

        text = soup.get_text().lower()
        domain_names = extract_domain_names(text)
        if domain_names:
            domain_count = Counter(domain_names)
            most_common_domain, _ = domain_count.most_common(1)[0]
            subdomain = ext.subdomain.lower()

            if most_common_domain in subdomain or most_common_domain in path.lower():
                features["EmbeddedBrandName"] = 1

            url_domain = tldextract.extract(url).domain

            if most_common_domain != url_domain:
                features["FrequentDomainNameMismatch"] = 1

        favicon = soup.find("link", rel="shortcut icon")
        if (
            favicon
            and urlparse(favicon["href"]).netloc
            and urlparse(favicon["href"]).netloc != hostname
        ):
            features["ExtFavicon"] = 1

        forms = soup.find_all("form", action=True)
        features["InsecureForms"] = sum(
            1
            for form in forms
            if form["action"].startswith("http:") or not urlparse(form["action"]).scheme
        )

        for form in forms:
            action = form["action"]
            action_url = urlparse(action)
            if not action_url.scheme:
                features["RelativeFormAction"] = 1
            elif action_url.netloc and action_url.netloc != hostname:
                features["ExtFormAction"] = 1
            if action in ["#", "about:blank", "", "javascript:true"]:
                features["AbnormalFormAction"] = 1
            if action.startswith("mailto:"):
                features["SubmitInfoToEmail"] = 1
            if form["action"].startswith("http:"):
                features["InsecureForms"] = 1

        scripts = soup.find_all("script")
        for script in scripts:
            if "onmouseover" in str(script):
                features["FakeLinkInStatusBar"] = 1
            if "oncontextmenu" in str(script):
                features["RightClickDisabled"] = 1
            if "window.open" in str(script):
                features["PopUpWindow"] = 1

        if soup.find_all(["iframe", "frame"]):
            features["IframeOrFrame"] = 1

        title = soup.find("title")
        if not title or not title.get_text(strip=True):
            features["MissingTitle"] = 1

        forms = soup.find_all("form")
        for form in forms:
            texts = form.find_all(string=True, recursive=False)
            images = form.find_all("img")
            if images and not any(text.strip() for text in texts):
                features["ImagesOnlyInForm"] = 1
                break

        resources = soup.find_all(["script", "link", "img"], src=True)
        external_resources = sum(
            1
            for res in resources
            if urlparse(res["src"]).netloc and urlparse(res["src"]).netloc != hostname
        )
        total_resources = len(resources)
        if total_resources > 0:
            features["PctExtResourceUrlsRT"] = (
                external_resources / total_resources
            ) * 100

        meta_script_links = soup.find_all(["meta", "script", "link"], src=True)
        external_meta_script_links = sum(
            1
            for tag in meta_script_links
            if "src" in tag.attrs
            and urlparse(tag["src"]).netloc
            and urlparse(tag["src"]).netloc != hostname
        )
        total_meta_script_links = len(meta_script_links)
        if total_meta_script_links > 0:
            features["ExtMetaScriptLinkRT"] = (
                external_meta_script_links / total_meta_script_links
            ) * 100

        for form in forms:
            action = form.get("action", "")
            if (
                urlparse(action).netloc
                and urlparse(action).netloc != hostname
                or action in ["about:blank", ""]
            ):
                features["AbnormalExtFormActionR"] = 1
                break

        links = soup.find_all("a", href=True)
        null_self_js_links = sum(
            1
            for link in links
            if link["href"].startswith("#")
            or link["href"].startswith("javascript:void(0)")
            or not link["href"]
        )
        total_links = len(links)
        if total_links > 0:
            features["PctExtNullSelfRedirectHyperlinksRT"] = (
                null_self_js_links / total_links
            ) * 100

    return {key: features[key] for key in feature_list}


feature_list = [
    "NumDots",
    "SubdomainLevel",
    "PathLevel",
    "UrlLength",
    "NumDash",
    "NumDashInHostname",
    "AtSymbol",
    "TildeSymbol",
    "NumUnderscore",
    "NumPercent",
    "NumQueryComponents",
    "NumAmpersand",
    "NumHash",
    "NumNumericChars",
    "NoHttps",
    "RandomString",
    "IpAddress",
    "DomainInSubdomains",
    "DomainInPaths",
    "HttpsInHostname",
    "HostnameLength",
    "PathLength",
    "QueryLength",
    "DoubleSlashInPath",
    "NumSensitiveWords",
    "EmbeddedBrandName",
    "PctExtHyperlinks",
    "PctExtResourceUrls",
    "ExtFavicon",
    "InsecureForms",
    "RelativeFormAction",
    "ExtFormAction",
    "AbnormalFormAction",
    "PctNullSelfRedirectHyperlinks",
    "FrequentDomainNameMismatch",
    "FakeLinkInStatusBar",
    "RightClickDisabled",
    "PopUpWindow",
    "SubmitInfoToEmail",
    "IframeOrFrame",
    "MissingTitle",
    "ImagesOnlyInForm",
    "SubdomainLevelRT",
    "UrlLengthRT",
    "PctExtResourceUrlsRT",
    "AbnormalExtFormActionR",
    "ExtMetaScriptLinkRT",
    "PctExtNullSelfRedirectHyperlinksRT",
]


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        return out


def predict_url_class(url, model):
    # Compute features from the URL
    features = compute_features(url, feature_list)
    features_df = pd.DataFrame([features])

    # using PyTorch, convert DataFrame to tensor
    inputs = torch.tensor(features_df.values.astype(np.float32))
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()


def classify_url(url):
    # Load the model
    nn_model = NeuralNet(
        input_size=len(feature_list), hidden_size=200, num_classes=2
    ).to(device)
    nn_model.load_state_dict(torch.load("nn_model.pth"))
    nn_model.eval()
    predicted_class = predict_url_class(url, nn_model)
    print("Predicted class:", "bad" if predicted_class == 1 else "good")
    message = (
        "The URL is suspicious. Be extremely careful when visiting suspicious website, and never disclose your personal credentials online."
        if predicted_class == 1
        else "The URL may be safe, but always exercise caution when visiting an unfamiliar website."
    )

    return message
