import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# ✅ Load your Excel file
df = pd.read_excel("2022-cs-606.xlsx")

# Urdu text cleaning function
def urdu_preprocessor(text):
    if not isinstance(text, str):
        return ""

    # Remove Urdu punctuation
    text = re.sub(r'[۔،؛؟!٭ء]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    try:
        tokens = word_tokenize(text)
        urdu_stopwords = set(stopwords.words('urdu')) if 'urdu' in stopwords.fileids() else set()
        cleaned_tokens = [token for token in tokens if token not in urdu_stopwords]
        return ' '.join(cleaned_tokens)
    except Exception:
        return text

# Apply cleaning function to the 'Urdu Sentence' column
df['cleaned_text'] = df['Urdu Sentence'].apply(urdu_preprocessor)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'],
    df['Sentiment'],
    test_size=0.2,
    random_state=42,
    stratify=df['Sentiment']
)

# Create a machine learning pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.7
    )),
    ('classifier', MultinomialNB())
])

# Train the model
print("Training the model...")
pipeline.fit(X_train, y_train)

# Evaluate the model's performance
y_pred = pipeline.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the model and vectorizer
joblib.dump(pipeline.named_steps['tfidf'], 'urdu_tfidf_vectorizer.pkl')
joblib.dump(pipeline.named_steps['classifier'], 'urdu_sentiment_nb_classifier.pkl')

print("\n Naive Bayes model training complete!")
print("Saved: urdu_tfidf_vectorizer.pkl")
print("Saved: urdu_sentiment_nb_classifier.pkl")

# Load the saved vectorizer and model
vectorizer = joblib.load('urdu_tfidf_vectorizer.pkl')
model = joblib.load('urdu_sentiment_nb_classifier.pkl')

# Urdu negations words list
urdu_negations = {
    "نہیں", "نہ", "مت", "ہرگز نہیں", "کبھی نہیں", "بالکل نہیں",
    "انکار", "ممنوع", "ناممکن", "غلط", "اجتناب", "عدم", "نا",
    "بے", "خلاف", "ممنوعہ", "رد", "منع", "منکر", "محروم",
    "مسترد", "نفی", "محفوظ نہیں", "غیر"
}

# Sentiment prediction function that checks for negations
def predict_sentiment(text):
    cleaned = urdu_preprocessor(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]

    # If the sentence contains a negative word, flip the sentiment
    if any(neg in text for neg in urdu_negations):
        prediction = 'N' if prediction == 'P' else 'P'

    return prediction

# Test exampl
test_urdu = "یہ فلم بہت بری نہیں تھی"
print(f"Test Sentence: {test_urdu}")
print("Predicted Sentiment:", predict_sentiment(test_urdu))
