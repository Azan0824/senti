import requests
import pandas as pd
import configparser
import re
from textblob import TextBlob
from wordcloud import WordCloud
import streamlit as st
import datetime, pytz

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"  # flags (iOS)
                           "]+", flags=re.UNICODE)

def get_rapidapi_headers():
    config = configparser.ConfigParser()
    config.read("config.ini")
    key = config["rapidapi"]["key"]
    host = config["rapidapi"]["host"]

    headers = {
        "X-RapidAPI-Key": key,
        "X-RapidAPI-Host": host
    }
    return headers

def cleanTxt(text):
    text = re.sub('@[A-Za-z0–9]+', '', text)
    text = re.sub('#', '', text)
    text = re.sub('RT[\s]+', '', text)
    text = re.sub('https?:\/\/\S+', '', text)
    text = re.sub("\n","",text)
    text = re.sub(":","",text)
    text = re.sub("_","",text)
    text = emoji_pattern.sub(r'', text)
    return text

def extract_mentions(text):
    text = re.findall("(@[A-Za-z0–9\d\w]+)", text)
    return text

def extract_hastag(text):
    text = re.findall("(#[A-Za-z0–9\d\w]+)", text)
    return text

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

@st.cache_data
def preprocessing_data(word_query, number_of_tweets, function_option):
    if function_option == "Twitter":
        headers = get_rapidapi_headers()
        all_tweets = []

        url = "https://twitter154.p.rapidapi.com/search/search?query=%23python&section=top&min_retweets=1&min_likes=1&limit=5&start_date=2022-01-01&language=en"
        querystring = {"query": word_query}

        response = requests.get(url, headers=headers, params=querystring)

        if response.status_code != 200:
            st.error(f"Failed to fetch tweets: {response.status_code}")
            return pd.DataFrame()

        tweets_json = response.json()

        for tweet in tweets_json.get("results", []):
            all_tweets.append(tweet.get("text", ""))

        if not all_tweets:
            return pd.DataFrame()

        data = pd.DataFrame(all_tweets, columns=["Tweets"])

        data["mentions"] = data["Tweets"].apply(extract_mentions)
        data["hastags"] = data["Tweets"].apply(extract_hastag)
        data['links'] = data['Tweets'].str.extract('(https?:\/\/\S+)', expand=False).str.strip()
        data['retweets'] = data['Tweets'].str.extract('(RT[\s@[A-Za-z0–9\d\w]+)', expand=False).str.strip()

        data['Tweets'] = data['Tweets'].apply(cleanTxt)
        discard = ["CNFTGiveaway", "GIVEAWAYPrizes", "Giveaway", "Airdrop", "GIVEAWAY", "makemoneyonline", "affiliatemarketing"]
        data = data[~data["Tweets"].str.contains('|'.join(discard), na=False)]

        data['Subjectivity'] = data['Tweets'].apply(getSubjectivity)
        data['Polarity'] = data['Tweets'].apply(getPolarity)
        data['Analysis'] = data['Polarity'].apply(getAnalysis)

        return data


def download_data(data, label):
    current_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
    current_time = f"{current_time.date()}.{current_time.hour}-{current_time.minute}-{current_time.second}"
    export_data = st.download_button(
        label=f"Download {label} data as CSV",
        data=data.to_csv(),
        file_name=f'{label}{current_time}.csv',
        mime='text/csv',
        help=f"When You Click On Download Button You can download your {label} CSV File"
    )
    return export_data

def analyse_mention(data):
    if "mentions" not in data.columns:
        # No mentions column at all
        return pd.DataFrame()

    if data["mentions"].dropna().empty:
        # Column exists but all values are NaN or empty
        return pd.DataFrame()

    # Safe to proceed with extraction
    try:
        mention_df = pd.DataFrame(data["mentions"].dropna().to_list()).add_prefix("mention_")
        mention_df = mention_df[mention_df["mention_username"].notnull()]
        top_mentions = mention_df["mention_username"].value_counts().head(10)
        fig = px.bar(top_mentions, x=top_mentions.index, y=top_mentions.values,
                     labels={"x": "Username", "y": "Count"},
                     title="Top 10 @Mentions in 100 tweets")
        st.plotly_chart(fig, use_container_width=True)
        return mention_df
    except Exception as e:
        st.warning(f"Could not generate mention chart: {e}")
        return pd.DataFrame()


def analyse_hastag(data):
    hastag_df = pd.DataFrame(data["hastags"].to_list()).add_prefix("hastag_")

    if hastag_df.empty:
        return pd.Series(dtype=int)

    cols_to_concat = [col for col in hastag_df.columns if col.startswith("hastag_")]

    hastag = pd.concat([hastag_df[col] for col in cols_to_concat], ignore_index=True)
    return hastag.value_counts().head(10)


def graph_sentiment(data):
    if "Analysis" not in data.columns or data.empty:
        return pd.DataFrame(columns=["Sentiment", "Count"])

    analys = data["Analysis"].value_counts().reset_index()
    analys.columns = ["Sentiment", "Count"]
    analys = analys.sort_values(by="Sentiment", ascending=False)
    return analys

