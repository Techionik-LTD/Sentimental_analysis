import pandas as pd
import re

# Read your data into a DataFrame
df = pd.read_csv("reviews.csv", encoding='latin-1')

# Remove rows where the 'Sentence' column is empty
df.dropna(subset=['Sentence'], inplace=True)


# Remove emojis and special characters
def remove_special_chars(text):
    # Remove emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    return text.strip()


# Apply remove_special_chars function to the 'Sentence' column
df['Sentence'] = df['Sentence'].apply(remove_special_chars)


# Remove extra spaces and join multiple lines into a single column
def remove_extra_spaces(text):
    return ' '.join(text.split())


# Apply remove_extra_spaces function to the 'Sentence' column
df['Sentence'] = df['Sentence'].apply(remove_extra_spaces)

# Remove empty rows
df = df[df['Sentence'] != '']

# Reset the index after dropping rows
df.reset_index(drop=True, inplace=True)

# Save the modified DataFrame back to a CSV file
df.to_csv("modified_data1.csv", index=False)
