import pandas as pd
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


def load_data():
    # Preferred format: CSV with columns label and message
    try:
        df = pd.read_csv("data/spam.csv", encoding="latin-1")
    except FileNotFoundError:
        # UCI original format is a tab-separated file named SMSSpamCollection
        df = pd.read_csv(
            "data/SMSSpamCollection",
            sep="\t",
            names=["label", "message"],
            encoding="utf-8"
        )
        return df

    # Keep only the first two useful columns if required
    if len(df.columns) >= 2:
        df = df.iloc[:, :2]
        df.columns = ["label", "message"]

    return df


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


df = load_data()
df = df.dropna()
df["label"] = df["label"].map({"ham": 0, "spam": 1})
df["clean_message"] = df["message"].apply(clean_text)

X = df["clean_message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

message = ["Congratulations! You won a free prize. Click now to claim."]
message_tfidf = vectorizer.transform(message)
prediction = model.predict(message_tfidf)

print("Prediction:", "SPAM" if prediction[0] == 1 else "HAM")
