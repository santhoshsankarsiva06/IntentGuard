# IntentGuard — Competition Submission

## Problem
A scam message can promise a scholarship or refund while its QR code requests a payment. Checking only the URL can miss this mismatch.

## Solution
IntentGuard is an offline Windows desktop prototype designed for Snapdragon-powered HP PCs. It accepts English, Tamil, and Thanglish messages with a URL or QR image. A locally stored scikit-learn model predicts message intent. URL/UPI parsing rules compare that intent with the target and explain suspicious mismatches.

## Architecture
Message → local scikit-learn intent classifier
URL or QR image → QR decoder → URL/UPI parser
Predicted intent + parsed target → mismatch rules → verdict and explanation

## Observed demo results
- English scholarship message + UPI payment: suspicious mismatch.
- Tamil scholarship message + UPI payment: suspicious mismatch.
- Thanglish scholarship message + UPI payment: suspicious mismatch.
- Workshop fee message + UPI payment: no mismatch found.

## Hardware and limitations
The intended target is a Snapdragon-powered HP PC. Snapdragon hardware performance and NPU acceleration have not been verified. The app does not open websites, verify payees, or guarantee that a link is safe.