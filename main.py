import nltk
import matplotlib.pyplot as plt

with open("moby_dick.txt", "r", encoding="utf-8") as f:
    moby_dick_text = f.read()

tokens = nltk.word_tokenize(moby_dick_text)

stop_words = set(nltk.corpus.stopwords.words("english"))
tokens = [token for token in tokens if token not in stop_words]

pos_tagged_tokens = nltk.pos_tag(tokens)

pos_counts = {}
for token, pos in pos_tagged_tokens:
    if pos in pos_counts:
        pos_counts[pos] += 1
    else:
        pos_counts[pos] = 1

print("The 5 most common POS and their total counts (frequency):")
for pos, count in sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{pos}: {count}")


lemmatizer = nltk.stem.WordNetLemmatizer()
top_20_lemmatized_tokens = [
    lemmatizer.lemmatize(token) for token in sorted(tokens)[:20]
]

plt.bar(pos_counts.keys(), pos_counts.values())
plt.xlabel("POS")
plt.ylabel("Frequency")
plt.title("POS Frequency Distribution")
plt.show()

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiment_analyzer = SentimentIntensityAnalyzer()
sentiment_scores = []
for sentence in nltk.sent_tokenize(moby_dick_text):
    sentiment_scores.append(sentiment_analyzer.polarity_scores(sentence)["compound"])

average_sentiment_score = sum(sentiment_scores) / len(sentiment_scores)

if average_sentiment_score > 0.05:
    overall_text_sentiment = "positive"
else:
    overall_text_sentiment = "negative"

print(f"Average sentiment score: {average_sentiment_score}")
print(f"Overall text sentiment: {overall_text_sentiment}")
