import csv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER sentiment analysis model
nltk.download('vader_lexicon')


def analyze_sentiment(sentence):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(sentence)

    # Determine the overall sentiment
    if sentiment_scores['compound'] >= 0.05:
        return "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"


# Open the CSV file for reading
with open('modified_data1.csv', 'r', encoding='utf-8') as f:
    # Create a CSV reader object
    csv_reader = csv.reader(f)
    # Skip the header row
    next(csv_reader)
    # Create a new CSV file for writing the sentiment analysis results
    with open('bert_sentimental_data.csv', 'w', newline='', encoding='utf-8') as output_file:
        # Create a CSV writer object
        csv_writer = csv.writer(output_file)
        # Write the header row
        csv_writer.writerow(['Sentence', 'Sentiment'])
        # Iterate over each row in the CSV file
        for row in csv_reader:
            # Extract the sentence from the row (assuming it's in the first column)
            sentence = row[0]
            # Perform sentiment analysis on the sentence
            sentiment = analyze_sentiment(sentence)
            # Write the sentiment analysis result to the output file
            csv_writer.writerow([sentence, sentiment])