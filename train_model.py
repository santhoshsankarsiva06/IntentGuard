"""Train the local intent classifier with the open-source scikit-learn platform."""
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

DATA = {
"receive_benefit": [
"Your scholarship money is ready", "You can receive the scholarship", "Claim your refund", "Cashback has been credited", "Collect your prize money", "The refund amount will be deposited", "Receive your stipend", "Your grant is available", "Scholarship amount is waiting", "You will get money back", "Collect your reward amount", "Your reimbursement is ready",
"உங்கள் உதவித்தொகை கிடைக்கும்", "உதவித்தொகை பணம் பெறுங்கள்", "பரிசுத் தொகை உங்களுக்கு கிடைத்துள்ளது", "பணம் திரும்ப கிடைக்கும்", "உங்கள் கணக்கில் பணம் வரவு வைக்கப்படும்", "இந்த பரிசை பெற்றுக் கொள்ளுங்கள்", "கல்லூரி உதவித்தொகையை பெறலாம்", "பணம் உங்கள் கணக்கிற்கு வரும்", "ரீஃபண்ட் தொகை கிடைக்கும்", "உதவித்தொகை தொகையை வாங்குங்கள்", "காசு திரும்ப கிடைக்கும்", "பரிசு வழங்கப்படும்",
"scholarship amount ungalukku kidaikkum", "scholarship panam receive pannunga", "unga account ku panam varum", "refund amount thirumbi varum", "cashback vandhuruchu collect pannunga", "prize money ungalukku kidaikum", "panam varum QR scan pannunga", "scholarship ready receive panna scan pannunga", "college grant panam vandhudum", "refund panam vaangikonga", "unga amount credit aagum", "reward money collect pannunga"],
"make_payment": [
"Pay the college fee today", "Please send money to this account", "Complete a payment of 500 rupees", "Transfer the registration fee", "Scan this QR code and pay", "Make a payment to confirm the seat", "Pay the workshop entry fee", "Pay the delivery charge", "Submit your admission payment", "Purchase the ticket using UPI", "The invoice is due please pay", "Send the amount to our UPI ID",
"கல்லூரி கட்டணம் செலுத்துங்கள்", "பணம் அனுப்புங்கள்", "இந்த QR மூலம் பணம் செலுத்துங்கள்", "பதிவு கட்டணம் செலுத்தவும்", "தொகையை இப்போது கட்டுங்கள்", "UPI மூலம் பணம் அனுப்பவும்", "நுழைவு கட்டணம் கட்ட வேண்டும்", "இந்த ரசீதை செலுத்துங்கள்", "பயிற்சி கட்டணம் செலுத்துங்கள்", "கட்டணத்தை உடனே செலுத்தவும்", "ஐநூறு ரூபாய் பணம் அனுப்பவும்", "இந்த தொகையை கட்டுங்கள்",
"college fee pay pannunga", "panam anuppunga", "intha QR la amount pay pannunga", "registration ku kaasu kattunga", "upi la transfer pannunga", "workshop fee seluthunga", "payment complete pannunga", "delivery charge pay pannu", "500 rooba anuppunga", "seat confirm panna fee pay pannunga", "amount kattitu receipt anuppunga", "QR scan panni panam kudunga"],
"share_credentials": [
"Enter your password to continue", "Share your OTP to verify the account", "Log in using your college credentials", "Type your bank PIN here", "Confirm your password on this form", "Submit the one time code", "Verify your identity with your login", "Tell us your verification code", "Enter the passcode sent to your phone", "Provide your username and password", "Update the account password", "Sign in and enter your OTP",
"உங்கள் கடவுச்சொல்லை உள்ளிடவும்", "OTP எண்ணை பகிரவும்", "கணக்கில் உள்நுழையவும்", "உங்கள் PIN எண்ணை தெரிவிக்கவும்", "சரிபார்ப்பு குறியீட்டை உள்ளிடவும்", "கடவுச்சொல்லை அனுப்பவும்", "வங்கிக் கணக்கு கடவுச்சொல் கொடுக்கவும்", "உள்நுழைவு தகவலை பதிவு செய்யவும்", "ஒரு முறை கடவுச்சொல் பகிரவும்", "உங்கள் ரகசிய எண்ணை உள்ளிடவும்", "OTP குறியீட்டை சொல்லுங்கள்", "கடவுச்சொல் மூலம் சரிபார்க்கவும்",
"password podunga", "otp sollunga", "college login details anuppunga", "bank pin share pannunga", "account verify panna password enter pannunga", "one time code anuppunga", "unga login id password kudu", "OTP code type pannunga", "upi pin enter pannu", "passcode share pannunga", "account la sign in pannunga", "verification code sollunga"],
"general_info": [
"The college library closes at five", "Tomorrow class starts at nine", "The workshop is on Friday", "Please attend the meeting", "The timetable has been updated", "Your project review is next week", "The seminar venue has changed", "Lunch is served at noon", "The registration desk is near the gate", "Here are the class notes", "The exam date has been announced", "Please read this notice",
"நாளை வகுப்பு காலை ஒன்பது மணிக்கு", "கல்லூரி நூலகம் இன்று திறந்திருக்கும்", "நாளை கூட்டம் நடைபெறும்", "வகுப்பு நேர அட்டவணை மாற்றப்பட்டுள்ளது", "தேர்வு தேதி அறிவிக்கப்பட்டது", "இந்த தகவலை படிக்கவும்", "பேருந்து நேரம் மாற்றப்பட்டது", "நிகழ்ச்சி நாளை தொடங்கும்", "ஆசிரியர் குறிப்புகளை பகிர்ந்தார்", "வெள்ளிக்கிழமை கருத்தரங்கு நடைபெறும்", "மாணவர்கள் கூட்டத்திற்கு வாருங்கள்", "நூலகம் ஐந்து மணிக்கு மூடும்",
"naalaiku class irukku", "meeting evening la nadakkum", "college bus time maariduchu", "workshop friday ku irukku", "review next week nadakkum", "notes share panniten", "timetable update aachu", "exam date announce pannanga", "seminar hall maathitanga", "teacher class edukkaranga", "library five mani close aagum", "event tomorrow start aagum"]}


def train():
    sentences = [sentence for examples in DATA.values() for sentence in examples]
    labels = [label for label, examples in DATA.items() for _ in examples]
    model = make_pipeline(CountVectorizer(analyzer="char", ngram_range=(2, 4), lowercase=True), MultinomialNB(alpha=1.0))
    model.fit(sentences, labels)
    path = Path(__file__).resolve().with_name("intent_model.joblib")
    joblib.dump(model, path)
    print(f"Saved {path.name} using scikit-learn; {len(sentences)} illustrative examples")


if __name__ == "__main__":
    train()
