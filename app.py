"""IntentGuard Day 2: local intent model with URL/QR mismatch checks."""
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from urllib.parse import parse_qs, unquote, urlsplit
from intent_model import predict

PAYMENT = ("pay", "payment", "fee", "transfer", "send money", "செலுத்த", "கட்டணம்", "panam anupp", "kattanam", "amount pay")
RECEIVE = ("receive", "scholarship", "refund", "cashback", "prize", "credited", "பெற", "உதவித்தொகை", "திரும்ப", "varum", "kidaikkum", "receive panna", "panam varum")
PASSWORD = ("password", "passcode", "otp", "login", "sign in", "கடவுச்சொல்", "உள்நுழை", "password podu", "otp sollu")
BRANDS = {"google": ("google.com",), "microsoft": ("microsoft.com", "live.com"), "sbi": ("sbi.co.in", "onlinesbi.sbi"), "amazon": ("amazon.in", "amazon.com")}


def contains(text, words):
    return any(word in text.lower() for word in words)


def parse_target(raw):
    raw = raw.strip()
    if not raw:
        return {"kind": "none", "detail": "No URL or QR content provided", "host": "", "payment": False, "credential": False, "flags": []}
    lower = raw.lower()
    if lower.startswith("upi://pay"):
        parts = urlsplit(raw)
        qs = parse_qs(parts.query)
        payee = qs.get("pa", ["not specified"])[0]
        amount = qs.get("am", ["not specified"])[0]
        return {"kind": "UPI payment", "detail": f"Payee: {payee}; amount: {amount}", "host": "", "payment": True, "credential": False, "flags": []}
    if not re.match(r"^https?://", raw, re.I):
        return {"kind": "other QR/text", "detail": raw[:200], "host": "", "payment": False, "credential": False, "flags": ["QR content is not an HTTP(S) URL or UPI payment request"]}
    try:
        parts = urlsplit(raw)
        host = (parts.hostname or "").lower().rstrip(".")
        if not host or "." not in host:
            raise ValueError("URL has no valid domain")
    except ValueError as exc:
        return {"kind": "invalid URL", "detail": str(exc), "host": "", "payment": False, "credential": False, "flags": ["Invalid URL"]}
    decoded = unquote(parts.path + "?" + parts.query).lower()
    flags = []
    if parts.scheme.lower() != "https":
        flags.append("URL does not use HTTPS")
    if parts.username or parts.password:
        flags.append("URL contains user-info before the domain")
    if host.startswith("xn--") or ".xn--" in host:
        flags.append("Domain uses internationalized characters; inspect carefully")
    credential = contains(decoded, ("login", "signin", "password", "verify", "account", "otp", "credential"))
    payment = contains(decoded, ("payment", "checkout", "pay-now", "paynow", "fee"))
    return {"kind": "URL", "detail": f"Destination domain: {host}", "host": host, "payment": payment, "credential": credential, "flags": flags}


def analyze(message, raw_target):
    message = message.strip()
    target = parse_target(raw_target)
    reasons = list(target["flags"])
    if not message and target["kind"] == "none":
        return "Enter a message or a URL/QR image.", target, []
    intent = predict(message)
    receive = bool(intent and intent["label"] == "receive_benefit")
    mentions_payment = bool(intent and intent["label"] == "make_payment") or contains(message, PAYMENT)
    if receive and target["payment"] and not mentions_payment:
        reasons.append("Message promises money/benefit, but destination requests payment")
    if message and target["credential"] and not contains(message, PASSWORD):
        reasons.append("Destination path suggests account verification/login, but message does not mention it")
    host = target["host"]
    for brand, domains in BRANDS.items():
        if brand in message.lower() and host and not any(host == d or host.endswith("." + d) for d in domains):
            reasons.append(f"Message mentions {brand}, but destination domain is {host} (check independently)")
            break
    if target["kind"] == "invalid URL":
        verdict = "Needs review"
    elif any("promises money" in r or "mentions" in r for r in reasons):
        verdict = "Suspicious mismatch"
    elif reasons:
        verdict = "Needs review"
    else:
        verdict = "No mismatch found (not a safety guarantee)"
    return verdict, target, reasons


def decode_qr(path):
    import cv2
    detector = cv2.QRCodeDetector()
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not open image")
    value, _, _ = detector.detectAndDecode(img)
    if not value:
        raise ValueError("No readable QR code found; try a clearer image")
    return value


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IntentGuard — Day 2 Local AI Prototype")
        self.geometry("760x560")
        self.columnconfigure(0, weight=1)
        ttk.Label(self, text="IntentGuard", font=("Segoe UI", 19, "bold")).grid(row=0, column=0, sticky="w", padx=20, pady=(18, 5))
        ttk.Label(self, text="Paste the full message, then paste its URL or upload a QR image.").grid(row=1, column=0, sticky="w", padx=20)
        ttk.Label(self, text="Message (English / Tamil / Thanglish):").grid(row=2, column=0, sticky="w", padx=20, pady=(18, 3))
        self.message = tk.Text(self, height=6, wrap="word")
        self.message.grid(row=3, column=0, sticky="ew", padx=20)
        ttk.Label(self, text="URL or decoded QR content:").grid(row=4, column=0, sticky="w", padx=20, pady=(15, 3))
        self.target = ttk.Entry(self)
        self.target.grid(row=5, column=0, sticky="ew", padx=20)
        buttons = ttk.Frame(self)
        buttons.grid(row=6, column=0, sticky="w", padx=20, pady=15)
        ttk.Button(buttons, text="Upload QR image", command=self.upload).pack(side="left", padx=(0, 10))
        ttk.Button(buttons, text="Analyze", command=self.run_analysis).pack(side="left")
        self.result = tk.Text(self, height=12, wrap="word", state="disabled")
        self.result.grid(row=7, column=0, sticky="nsew", padx=20, pady=(0, 8))
        self.rowconfigure(7, weight=1)
        ttk.Label(self, text="Local trained intent model + URL/QR rules. Does not prove a destination is safe.").grid(row=8, column=0, sticky="w", padx=20, pady=(0, 15))

    def upload(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            try:
                value = decode_qr(path)
                self.target.delete(0, "end")
                self.target.insert(0, value)
            except ValueError as exc:
                messagebox.showerror("QR image", str(exc))

    def run_analysis(self):
        verdict, target, reasons = analyze(self.message.get("1.0", "end"), self.target.get())
        intent = predict(self.message.get("1.0", "end"))
        model_info = ("Message intent (local AI): " + intent["display"] + "\n" if intent else "Message intent (local AI): no message\n")
        output = f"VERDICT: {verdict}\n\n{model_info}Decoded target: {target['kind']}\n{target['detail']}\n\n"
        output += "REASONS:\n" + ("\n".join("• " + r for r in reasons) if reasons else "• No rule matched; manually verify before acting.")
        self.result.config(state="normal")
        self.result.delete("1.0", "end")
        self.result.insert("1.0", output)
        self.result.config(state="disabled")


if __name__ == "__main__":
    App().mainloop()
