import nltk
import matplotlib.pyplot as plt

# Read the Moby Dick file from the Gutenberg dataset
with open("moby_dick.txt", "r", encoding="utf-8") as f:
    moby_dick_text = f.read()

# Tokenize the text
tokens = nltk.word_tokenize(moby_dick_text)

# Filter out stop words
stop_words = set(nltk.corpus.stopwords.words("english"))
tokens = [token for token in tokens if token not in stop_words]

# Tag the parts of speech (POS) for each word
pos_tagged_tokens = nltk.pos_tag(tokens)

# Count the frequency of each POS
pos_counts = {}
for token, pos in pos_tagged_tokens:
    if pos in pos_counts:
        pos_counts[pos] += 1
    else:
        pos_counts[pos] = 1

# Display the 5 most common POS and their total counts (frequency)
print("The 5 most common POS and their total counts (frequency):")
for pos, count in sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{pos}: {count}")

# Lemmatize the top 20 tokens
lemmatizer = nltk.stem.WordNetLemmatizer()
top_20_lemmatized_tokens = [
    lemmatizer.lemmatize(token) for token in sorted(tokens)[:20]
]

# Plot a bar chart to visualize the frequency of POS
plt.bar(pos_counts.keys(), pos_counts.values())
plt.xlabel("POS")
plt.ylabel("Frequency")
plt.title("POS Frequency Distribution")
plt.show()

# Perform sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiment_analyzer = SentimentIntensityAnalyzer()
sentiment_scores = []
for sentence in nltk.sent_tokenize(moby_dick_text):
    sentiment_scores.append(sentiment_analyzer.polarity_scores(sentence)["compound"])

# Calculate the average sentiment score
average_sentiment_score = sum(sentiment_scores) / len(sentiment_scores)

# Determine if the overall text sentiment is positive or negative
if average_sentiment_score > 0.05:
    overall_text_sentiment = "positive"
else:
    overall_text_sentiment = "negative"

# Display the average sentiment score and state if the overall text sentiment is positive or negative
print(f"Average sentiment score: {average_sentiment_score}")
print(f"Overall text sentiment: {overall_text_sentiment}")
