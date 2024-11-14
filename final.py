import streamlit as st
import mysql.connector
from mysql.connector import Error
import pandas as pd
from dotenv import load_dotenv
import os
from textblob import TextBlob
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import praw
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import spacy
from spacy.matcher import Matcher
from collections import Counter
from openai import OpenAI
import subprocess
import sys
from spacy.cli import download

nlp = spacy.load("en_core_web_sm")

# Retrieve the environment variables
DB_HOST = st.secrets["MYSQL_HOST"]
DB_USER = st.secrets["MYSQL_USER"]
DB_PASSWORD = st.secrets["MYSQL_PASSWORD"]
DB_NAME = st.secrets["MYSQL_DATABASE"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
reddit = praw.Reddit(
    client_id=st.secrets["REDDIT_CLIENT_ID"],      
    client_secret=st.secrets["REDDIT_CLIENT_SECRET"], 
    user_agent='SchoolReviews'
)
current_date = datetime.now()
scrape_year = current_date.year
scrape_month = current_date.month
scrape_date = current_date.day

# Define keywords
keywords = ["Nanyang Business School", "National University Singapore"]

# Function to scrape Mothership
def scrape_mothership():
    url = 'https://mothership.sg/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    data1 = []
    for article in soup.find_all('article'):
        title = article.find('h2').text.strip() if article.find('h2') else 'No title'
        link = article.find('a')['href'] if article.find('a') else 'No link'
        date = article.find('time')['datetime'] if article.find('time') else 'No date'
        combined_date = f"{scrape_year}-{scrape_month:02d}-{scrape_date:02d}"
        data1.append({
            "date": combined_date,
            'year': scrape_year,
            'month': scrape_month,
            'day': scrape_date,
            "link": link,
            'content': title,
            'source' : "Straits Times Multi Media"
        })

    return pd.DataFrame(data1)

# Function to scrape CNA
def scrape_cna():
    base_url = 'https://www.channelnewsasia.com'
    url = 'https://www.channelnewsasia.com/'

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    data2 = []
    article_classes = ['h6__link list-object__heading-link', 'feature-card__heading-link']

    for article_class in article_classes:
        for article in soup.find_all('a', class_=article_class):
            title = article.text.strip() if article.text else 'No title'
            link = base_url + article['href'] if 'href' in article.attrs else 'No link'
            combined_date = f"{scrape_year}-{scrape_month:02d}-{scrape_date:02d}"
            data2.append({
                "date": combined_date,
                'year': scrape_year,
                'month': scrape_month,
                'day': scrape_date,
                "link": link,
                'content': title,
                'source' : "Straits Times Multi Media"
            })

    return pd.DataFrame(data2)

# Function to scrape Business Times
def scrape_biztimes():
    url = 'https://www.businesstimes.com.sg/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    data3 = []
    base_url = "https://www.businesstimes.com.sg"

    for article in soup.find_all('a', class_='inherit word-break hover:underline text-gray-850'):
        title = article['title'] if 'title' in article.attrs else 'No title'
        link = base_url + article['href'] if 'href' in article.attrs else 'No link'
        combined_date = f"{scrape_year}-{scrape_month:02d}-{scrape_date:02d}"
        data3.append({
            "date": combined_date,
            'year': scrape_year,
            'month': scrape_month,
            'day': scrape_date,
            "link": link,
            'content': title,
            'source' : "Straits Times Multi Media"
        })

    return pd.DataFrame(data3)

# Function to scrape Straits Times
def scrape_straitstimes():
    base_url = 'https://www.straitstimes.com'
    url = 'https://www.straitstimes.com/'

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    data4 = []
    for article in soup.find_all('a', class_='stretched-link'):
        title = article.text.strip() if article.text else 'No title'
        link = base_url + article['href'] if 'href' in article.attrs else 'No link'
        combined_date = f"{scrape_year}-{scrape_month:02d}-{scrape_date:02d}"
        data4.append({
            "date": combined_date,
            'year': scrape_year,
            'month': scrape_month,
            'day': scrape_date,
            "link": link,
            'content': title,
            'source' : "Straits Times Multi Media"

        })

    return pd.DataFrame(data4)

# Function to scrape Multimedia from Straits Times
def scrape_multimedia():
    base_url = 'https://www.straitstimes.com'
    url = 'https://www.straitstimes.com/multimedia'

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    data5 = []
    for article in soup.find_all('a', class_='stretched-link'):
        card_title = article.find_previous('h5', class_='card-title')
        title = card_title.text.strip() if card_title else 'No title'
        link = base_url + article['href'] if 'href' in article.attrs else 'No link'
        combined_date = f"{scrape_year}-{scrape_month:02d}-{scrape_date:02d}"

        data5.append({
            "date": combined_date,
            'year': scrape_year,
            'month': scrape_month,
            'day': scrape_date,
            "link": link,
            'content': title,
            'source' : "Straits Times Multi Media"
        })

    return pd.DataFrame(data5)

@st.cache_data
def scrape_all_sources():
    dataframes = [
        scrape_mothership(),
        scrape_cna(),
        scrape_biztimes(),
        scrape_straitstimes(),
        scrape_multimedia()
    ]
    if dataframe:
        return pd.concat(dataframes, ignore_index=True)
    else: 
        st.write("There is no current news on NTU and NUS")

def filter_news_by_keywords(data, keywords):
    return data[data['content'].str.contains('|'.join(keywords), case=False)]
    

def analyze_sentiment_aspects(df):
    sentiments = []
    polarities = []
    subjectivities = []
    magnitudes = []
    aspects_list = []

    for _, row in df.iterrows():
        comment = row["content"]
        blob = TextBlob(comment)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        magnitude = abs(polarity)
        sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
        sentiments.append(sentiment)
        polarities.append(polarity)
        subjectivities.append(subjectivity)
        magnitudes.append(magnitude)

        # Aspect mining using spaCy
        aspects = extract_aspects(comment)
        aspects_list.append(aspects)
    
    # Add the results to the DataFrame
    df["sentiment"] = sentiments
    df["polarity"] = polarities
    df["subjectivity"] = subjectivities
    df["magnitude"] = magnitudes
    df["aspects"] = aspects_list

    return df
def extract_aspects(text):
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)

    pattern = [
        {"POS": "ADJ", "OP": "*"},  
        {"POS": "NOUN"}             
    ]
    matcher.add("AspectMining", [pattern])

    aspects = []
    
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        aspects.append(span.text)  

    return aspects


def homepage():
    
    st.title("Whats Poppin on the News")
    st.write("News sources are scraped from Mothership, Straits Times, CNA and Business Times")
    
    if 'data' not in st.session_state:
        st.session_state['data'] = scrape_all_sources()

    # Filter the data for NTU and NUS
    filtered_data = filter_news_by_keywords(st.session_state['data'], keywords)

    if not filtered_data.empty:
        st.write("### Recent News")
        # Display filtered data in the specified format
        for index, row in filtered_data.iterrows():
            st.write(f"**Date:** {row['date']}")
            st.write(f"**Source:** {row['source']}")
            st.write(f"**Content:** {row['content']}")
            st.write(f"Link to post: {row['link']}")
            st.write("---") 
@st.cache_data
def load_and_analyze_data(table_name):
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        if conn.is_connected():
            query = f"SELECT * FROM {table_name};"
            df = pd.read_sql(query, conn)
            conn.close()

            
            if 'content' in df.columns:   
                sentiments = []
                polarities = []
                subjectivities = []
                magnitudes = []
                aspects_list = []

                for _, row in df.iterrows():
                    comment = row["content"]
                    blob = TextBlob(comment)
                    polarity = blob.sentiment.polarity
                    subjectivity = blob.sentiment.subjectivity
                    magnitude = abs(polarity)
                    sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
                    
                    sentiments.append(sentiment)
                    polarities.append(polarity)
                    subjectivities.append(subjectivity)
                    magnitudes.append(magnitude)

                    # Aspect mining using spaCy
                    aspects = extract_aspects(comment)
                    aspects_list.append(aspects)
                
                # Add the results to the DataFrame
                df["sentiment"] = sentiments
                df["polarity"] = polarities
                df["subjectivity"] = subjectivities
                df["magnitude"] = magnitudes
                df["aspects"] = aspects_list

                return df
            else:
                st.error("The 'content' column is missing in the table.")
                return None
    except mysql.connector.Error as e:
        st.error(f"Error connecting to the database: {e}")
        return None
@st.cache_data    
def connect_to_rds():
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        if conn.is_connected():
            st.success("Successfully connected to the database")
            return conn
        else:
            st.error("Failed to connect to the database")
            return None
    except Error as e:
        st.error(f"Error connecting to MySQL platform: {e}")
        return None
    
def refresh_data():
    conn = connect_to_rds()
    if conn:
        scrape_reddit(conn, 'NTU', ["Business", "NBS"], 'NTU_dashboard', limit=20)  
        scrape_reddit(conn, 'nus', ['NUS business', 'nus biz'], 'NUS_dashboard', limit=20)

        conn.close()
        st.success("Data refreshed successfully!")
    else:
        st.error("Could not refresh data due to a database connection issue.")
# Function to generate sentiment distribution graph
def sentiment_distribution_plot(df, title):
    if 'polarity' in df.columns:
        fig = px.histogram(df, x='polarity', nbins=20, title=title)
        return fig
    else:
        st.warning("Sentiment data is missing.")
        return None

# Function to generate word cloud
def word_cloud_plot(df, title):
    text = ' '.join(df['content'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig
@st.cache_data
def fetch_nus_data():
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        if conn.is_connected():
            
            query = """
                SELECT year, month, day, post, content, comments 
                FROM NUS_dashboard_reddit
                ORDER BY year DESC, month DESC, day DESC;
            """
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        else:
            st.error("Failed to connect to the database")
            return None
            
    except Error as e:
        st.error(f"Error connecting to the database: {e}")
        return None
    

def sentiment_line_plot(df, title):
    if 'Year-Month' not in df.columns:
        df['Year-Month'] = pd.to_datetime(df[['year', 'month']].assign(day=1)).dt.strftime('%Y-%m')

   
    sentiment_df = df.groupby('Year-Month')['polarity'].mean().reset_index()

   
    fig = px.scatter(
        sentiment_df, 
        x='Year-Month', 
        y='polarity', 
        title=title,
        labels={'Polarity': 'Average Sentiment', 'Year-Month': 'Month-Year'},
        
    )
    return fig

def sentiment_pie_chart(df, title):
    if 'polarity' in df.columns:
        sentiment_labels = ['Positive' if x > 0 else 'Negative' for x in df['polarity']]
        df['Sentiment Label'] = sentiment_labels
        fig = px.pie(df, names='Sentiment Label', title=title)
        return fig
    else:
        st.warning("Polarity data is missing for pie chart.")
        return None


def display_metrics(df):
    total_reviews = df.shape[0]
    positive_reviews = df[df['polarity'] > 0].shape[0]
    negative_reviews = total_reviews - positive_reviews

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Reviews", value=total_reviews)
    with col2:
        st.metric(label="Positive Reviews", value=f"{positive_reviews} ({(positive_reviews / total_reviews) * 100:.2f}%)")
    with col3:
        st.metric(label="Negative Reviews", value=f"{negative_reviews} ({(negative_reviews / total_reviews) * 100:.2f}%)")

# Function for filtering data
def filter_data(df, selected_year, selected_month, selected_source, selected_sentiment):
    if selected_year != 'All':
        df = df[df['year'] == selected_year]
    if selected_month != 'All':
        df = df[df['month'] == selected_month]
    if selected_source != 'All':
        df = df[df['source'] == selected_source]
    if selected_sentiment != 'All':
        if selected_sentiment == 'Positive':
            df = df[df['polarity'] > 0]
        elif selected_sentiment == 'Negative':
            df = df[df['polarity'] <= 0]
    return df

def insert_entry(conn, entry, table_name):
    sql_check = f'''SELECT COUNT(*) FROM {table_name} WHERE content = %s'''
    sql_insert = f'''INSERT INTO {table_name} (year, month, day, content, source)
                     VALUES (%s, %s, %s, %s, %s)'''
    try:
        cursor = conn.cursor()
        cursor.execute(sql_check, (entry[3],))
        result = cursor.fetchone()

        if result[0] == 0:  # If no duplicate found, proceed to insert
            cursor.execute(sql_insert, entry)
            conn.commit()
            print("Entry inserted successfully!")
        else:
            print("Duplicate entry found. Skipping insertion.")
    except Error as e:
        print(f"Error inserting entry: '{e}'")

def scrape_reddit(conn, subreddit_name, keyword, table_name, limit=20):
    subreddit = reddit.subreddit(subreddit_name)
    for submission in subreddit.search(keyword, limit=limit):
        post_date = datetime.fromtimestamp(submission.created_utc)
        year = post_date.year
        month = post_date.month
        day = post_date.day
        post_body = submission.selftext
        source = 'Reddit'

        post_entry = (year, month, day, post_body, source)
        insert_entry(conn, post_entry, table_name)

        
        submission.comments.replace_more(limit=None)  
        for comment in submission.comments.list():
            comment_body = comment.body.strip()
            if len(comment_body.split()) >= 7:  
                comment_date = datetime.fromtimestamp(comment.created_utc)
                comment_year = comment_date.year
                comment_month = comment_date.month
                comment_day = comment_date.day

                # Insert the comment as a separate entry with the source
                comment_entry = (comment_year, comment_month, comment_day, comment_body, source)
                insert_entry(conn, comment_entry, table_name)


def ntu_reviews_page(df):
    st.title("School Reviews Dashboard")
    if st.button("Refresh Data(Approximately 3 Mins)"):
        refresh_data()
    st.subheader("NTU Reviews")
    
    # Sidebar filters specific to NTU page
    years = ['All'] + sorted(df['year'].unique().tolist())
    months = ['All'] + sorted(df['month'].unique().tolist())
    sources = ['All'] + sorted(df['source'].unique().tolist())
    sentiments = ['All', 'Positive', 'Negative']

    selected_year = st.sidebar.selectbox("Select Year", years)
    selected_month = st.sidebar.selectbox("Select Month", months)
    selected_source = st.sidebar.selectbox("Select Source", sources)
    selected_sentiment = st.sidebar.selectbox("Select Sentiment", sentiments)

    # Apply filters to NTU dataset
    filtered_ntu = filter_data(df, selected_year, selected_month, selected_source, selected_sentiment)

    # Display metrics and visualizations
    display_metrics(filtered_ntu)

    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        fig1 = sentiment_distribution_plot(filtered_ntu, "Sentiment Distribution for NTU Reviews")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = word_cloud_plot(filtered_ntu, "Word Cloud of NTU Reviews")
        st.pyplot(fig2)

    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        fig3 = sentiment_line_plot(filtered_ntu, "Sentiment Over Time for NTU Reviews")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = sentiment_pie_chart(filtered_ntu, "Sentiment Breakdown for NTU Reviews")
        st.plotly_chart(fig4, use_container_width=True)

def nus_reviews_page(df):
    st.title("School Reviews Dashboard")
    if st.button("Refresh Data(Approximately 3 Mins)"):
        refresh_data()
    st.subheader("NUS Reviews")
    
    # Sidebar filters specific to NUS page
    years = ['All'] + sorted(df['year'].unique().tolist())
    months = ['All'] + sorted(df['month'].unique().tolist())
    sources = ['All'] + sorted(df['source'].unique().tolist())
    sentiments = ['All', 'Positive', 'Negative']

    selected_year = st.sidebar.selectbox("Select Year", years)
    selected_month = st.sidebar.selectbox("Select Month", months)
    selected_source = st.sidebar.selectbox("Select Source", sources)
    selected_sentiment = st.sidebar.selectbox("Select Sentiment", sentiments)

    # Apply filters to NUS dataset
    filtered_nus = filter_data(df, selected_year, selected_month, selected_source, selected_sentiment)

    # Display metrics and visualizations
    display_metrics(filtered_nus)

    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        fig1 = sentiment_distribution_plot(filtered_nus, "Sentiment Distribution for NUS Reviews")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = word_cloud_plot(filtered_nus, "Word Cloud of NUS Reviews")
        st.pyplot(fig2)

    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        fig3 = sentiment_line_plot(filtered_nus, "Sentiment Over Time for NUS Reviews")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = sentiment_pie_chart(filtered_nus, "Sentiment Breakdown for NUS Reviews")
        st.plotly_chart(fig4, use_container_width=True)

def stalking_nus_page():
    st.title("Stalking NUS")
    if st.button("Refresh Data(Approximately 3 Mins)"):
        refresh_data()

    # Fetch data from the database
    df = fetch_nus_data()

    if df is not None and not df.empty:
        # Iterate through each post and display it with its comments
        for index, row in df.iterrows():
            st.subheader(f"Post Date: {row['year']}-{row['month']}-{row['day']}")
            st.markdown(f"**Post:** {row['post']}")
            st.markdown(f"**Content:** {row['content']}")

            # Split the combined comments using " | " as the delimiter
            comments = row['comments'].split(" | ")

            st.markdown("**Comments:**")
            for comment in comments:
                st.write(f"- {comment.strip()}")
    else:
        st.warning("No data available in the NUS dashboard.")

@st.cache_data
def extract_most_frequent_positive_nouns(df):
    # Filter for positive comments
    positive_comments = df[df['polarity'] > 0]['content']
    noun_counter = Counter()
    
    # Process each positive comment
    for comment in positive_comments:
        doc = nlp(comment)
        nouns = [token.text for token in doc if token.pos_ == "NOUN"]
        noun_counter.update(nouns)
    
    # Get the most common nouns
    most_common_nouns = noun_counter.most_common(5)
    return most_common_nouns

def comparison_page(df_ntu, df_nus):
    st.subheader("Comparison of NTU and NUS Reviews")

    # Calculate total, positive, negative, and neutral reviews for each school
    ntu_total = df_ntu.shape[0]
    ntu_positive = df_ntu[df_ntu['polarity'] > 0].shape[0]
    ntu_negative = df_ntu[df_ntu['polarity'] < 0].shape[0]
    ntu_neutral = ntu_total - ntu_positive - ntu_negative

    nus_total = df_nus.shape[0]
    nus_positive = df_nus[df_nus['polarity'] > 0].shape[0]
    nus_negative = nus_total - nus_positive - ntu_neutral
    nus_neutral = nus_total - nus_positive - nus_negative

    # Combine NTU and NUS data for plotting
    comparison_df = pd.DataFrame({
        'School': ['NTU', 'NUS'],
        'Positive Reviews': [ntu_positive, nus_positive],
        'Negative Reviews': [ntu_negative, nus_negative],
        'Neutral Reviews': [ntu_neutral, nus_neutral],
        'Total Reviews': [ntu_total, nus_total]
    })

    # Full-width layout for better visibility
    st.subheader("Summary of Reviews")

    # Display total, positive, negative, and neutral reviews side by side
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total Reviews (NTU)", value=ntu_total)
        st.metric(label="Total Positive Reviews (NTU)", value=ntu_positive)
        st.metric(label="Total Negative Reviews (NTU)", value=ntu_negative)
        st.metric(label="Total Neutral Reviews (NTU)", value=ntu_neutral)
    with col2:
        st.metric(label="Total Reviews (NUS)", value=nus_total)
        st.metric(label="Total Positive Reviews (NUS)", value=nus_positive)
        st.metric(label="Total Negative Reviews (NUS)", value=nus_negative)
        st.metric(label="Total Neutral Reviews (NUS)", value=nus_neutral)

    # Display most frequent positive nouns
    st.subheader("Most Frequent Positive Nouns")
    ntu_positive_nouns = extract_most_frequent_positive_nouns(df_ntu)
    nus_positive_nouns = extract_most_frequent_positive_nouns(df_nus)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**NTU Most Frequent Positive Nouns**")
        for noun, count in ntu_positive_nouns:
            st.write(f"{noun}: {count}")
    with col2:
        st.write("**NUS Most Frequent Positive Nouns**")
        for noun, count in nus_positive_nouns:
            st.write(f"{noun}: {count}")
    # Pie chart for NTU sentiment distribution
    st.subheader("Sentiment Distribution for NTU Reviews")
    ntu_sentiment_counts = pd.Series([ntu_positive, ntu_neutral, ntu_negative], 
                                     index=['Positive', 'Neutral', 'Negative'])
    fig_ntu_pie = px.pie(
        ntu_sentiment_counts,
        values=ntu_sentiment_counts.values,
        names=ntu_sentiment_counts.index,
        title="NTU Sentiment Distribution",
        labels={'value': 'Number of Reviews'}
    )
    st.plotly_chart(fig_ntu_pie, use_container_width=True)

    # Pie chart for NUS sentiment distribution
    st.subheader("Sentiment Distribution for NUS Reviews")
    nus_sentiment_counts = pd.Series([nus_positive, nus_neutral, nus_negative], 
                                     index=['Positive', 'Neutral', 'Negative'])
    fig_nus_pie = px.pie(
        nus_sentiment_counts,
        values=nus_sentiment_counts.values,
        names=nus_sentiment_counts.index,
        title="NUS Sentiment Distribution",
        labels={'value': 'Number of Reviews'}
    )
    st.plotly_chart(fig_nus_pie, use_container_width=True)
    # Bar graph comparing positive, negative, and neutral reviews
    st.subheader("Bar Graph Comparison of Reviews")
    fig = px.bar(
        comparison_df,
        x='School',
        y=['Positive Reviews', 'Negative Reviews', 'Neutral Reviews'],
        title="Comparison of Positive, Negative, and Neutral Reviews",
        barmode='group',
        labels={'value': 'Number of Reviews'},
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Time-based analysis (Total Reviews and Positive Reviews over time - monthly)
    st.subheader("Monthly Review Trends for NTU and NUS")

    # Group by Year-Month and count total and positive reviews for each month
    df_ntu['Year-Month'] = df_ntu['year'].astype(str) + '-' + df_ntu['month'].astype(str)
    df_nus['Year-Month'] = df_nus['year'].astype(str) + '-' + df_nus['month'].astype(str)

    # NTU total reviews and positive reviews over time
    ntu_monthly = df_ntu.groupby('Year-Month').size().reset_index(name='Total Reviews')
    ntu_positive_monthly = df_ntu[df_ntu['polarity'] > 0].groupby('Year-Month').size().reset_index(name='Positive Reviews')

    # NUS total reviews and positive reviews over time
    nus_monthly = df_nus.groupby('Year-Month').size().reset_index(name='Total Reviews')
    nus_positive_monthly = df_nus[df_nus['polarity'] > 0].groupby('Year-Month').size().reset_index(name='Positive Reviews')

    # Df Handling
    total_reviews_df = pd.merge(ntu_monthly, nus_monthly, on='Year-Month', suffixes=('_NTU', '_NUS'))
    total_reviews_df = total_reviews_df.sort_values('Year-Month').reset_index(drop=True)
    total_reviews_df_melted = total_reviews_df.melt(
        id_vars=['Year-Month'], 
        value_vars=['Total Reviews_NTU', 'Total Reviews_NUS'],
        var_name='School', 
        value_name='Total Reviews'
    )
    
    st.subheader("Total Reviews Over Time")
    fig_total_reviews = px.scatter(
        total_reviews_df_melted,
        x='Year-Month', 
        y='Total Reviews',
        color='School',
        labels={'Total Reviews': 'Total Reviews', 'Year-Month': 'Month'},
        title="Total Reviews Over Time (NTU vs NUS)"
    )
    st.plotly_chart(fig_total_reviews, use_container_width=True)

    # Merge NTU and NUS data for Positive Reviews
    positive_reviews_df = pd.merge(ntu_positive_monthly, nus_positive_monthly, on='Year-Month', suffixes=('_NTU', '_NUS'))
    positive_reviews_df = positive_reviews_df.sort_values('Year-Month').reset_index(drop=True)
    positive_reviews_df_melted = positive_reviews_df.melt(
        id_vars=['Year-Month'], 
        value_vars=['Positive Reviews_NTU', 'Positive Reviews_NUS'],
        var_name='School', 
        value_name='Positive Reviews'
    )

    # Line chart for positive reviews over time
    st.subheader("Positive Reviews Over Time")
    fig_positive_reviews = px.scatter(
        positive_reviews_df_melted,
        x='Year-Month', 
        y='Positive Reviews',
        color='School',
        labels={'Positive Reviews': 'Positive Reviews', 'Year-Month': 'Month'},
        title="Positive Reviews Over Time (NTU vs NUS)"
    )
    st.plotly_chart(fig_positive_reviews, use_container_width=True)
@st.cache_data
def load_data_from_db():
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        if conn.is_connected():
            query = "SELECT * FROM NUS_dashboard_reddit;"  
            df = pd.read_sql(query, conn)
            conn.close()
            return df
    except Error as e:
        st.error(f"Error connecting to the database: {e}")
        return None
    
def generate_action_plan_gpt3(title, content, selected_comments):

    client = OpenAI(
    api_key = OPENAI_API_KEY
)
    # Combine the selected comments into a single string
    comments = "\n".join(selected_comments) if selected_comments else "No relevant comments selected."
    
    # Create a structured prompt with the selected comments
    prompt = (
        f"Post Title: {title}\n"
        f"Content: {content}\n"
        f"Comments: {comments}\n\n"
        "As a school dean, generate a structured action plan based on the post and comments. "
        "The action plan should address any issues raised and provide clear, actionable steps for NBS. "
        "Please include the following sections:\n\n"
        "1. **Goals**: Outline the main objectives.\n"
        "2. **Action Steps**: List specific actions to address each objective.\n"
        "3. **Timeline**: Suggest a timeline for each action.\n"
        "4. **Resources Needed**: Identify any resources or support required.\n"
        "5. **Expected Outcomes**: Describe the anticipated results from implementing the plan."
    )

    # Use OpenAI API to generate a response
    response =client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You are a school dean helping to generate structured and actionable plans."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7
    )
    
    # Extract and return the generated action plan text
    action_plan = response.choices[0].message
    return action_plan

def ai_page():
    st.title("Action Plan Generation for NBS")

    # Load the data from the database
    df = load_data_from_db()
    
    if df is not None:
        # Search bar to filter posts
        search_term = st.text_input("Search for posts related to NUS/NBS:")
        
        # Filter the DataFrame based on the search term
        if search_term:
            filtered_df = df[df['post'].str.contains(search_term, case=False, na=False)]
        else:
            filtered_df = df

        # Display the filtered posts and allow selection
        post_selection = st.selectbox("Select a post:", filtered_df['post'])

        # Show the selected post details
        if post_selection:
            selected_post = filtered_df[filtered_df['post'] == post_selection].iloc[0]
            st.subheader("Selected Post")
            st.write(selected_post['content'])

            # Display comments associated with the post
            st.subheader("Comments")
            comments = selected_post['comments']
            if comments:
                comment_list = comments.split(" | ")
                selected_comments = []
                
                # Let users select comments to include
                st.write("Select comments to include in the action plan:")
                for idx, comment in enumerate(comment_list, 1):
                    if st.checkbox(f"Comment {idx}: {comment}", key=f"comment_{idx}"):
                        selected_comments.append(comment)

                # Combine selected comments for the action plan generation
                combined_comments = "\n".join(selected_comments) if selected_comments else "No comments selected."
            else:
                st.write("No comments available for this post.")
                combined_comments = "No comments available."

            # Generate action plan button
            if st.button("Generate Action Plan"):
                with st.spinner("Generating action plan..."):
                    action_plan = generate_action_plan_gpt3(
                        selected_post['post'], 
                        selected_post['content'], 
                        selected_comments
                    )
                    if action_plan:
                        st.subheader("Action Plan")
                        st.write(action_plan)
                    else:
                        st.error("Failed to generate an action plan.")
    else:
        st.error("No data available. Ensure the database connection is correct.")
    
    
def main():
    page = st.sidebar.selectbox("Select a Page", ["Recent News","NTU Reviews", "NUS Reviews", "Comparison","Stalking NUS","AI Generator"])

    # Load data
    df_ntu = load_and_analyze_data("NTU_dashboard")
    df_nus = load_and_analyze_data("NUS_dashboard")

        # Navigation logic for pages
    if page == "Recent News":
        homepage()
    elif page == "NTU Reviews":
        ntu_reviews_page(df_ntu)
    elif page == "NUS Reviews":
        nus_reviews_page(df_nus)
    elif page == "Comparison":
        comparison_page(df_ntu, df_nus)
    elif page == "Stalking NUS":
        stalking_nus_page()
    elif page == "AI Generator":
        ai_page()
    
if __name__ == "__main__":
    main()
