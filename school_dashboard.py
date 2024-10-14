import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO

@st.cache_data
def load_data():
    # Use the raw GitHub URL for the Excel file
    github_url = 'https://raw.githubusercontent.com/SantosJustinian/ThreadZero01/main/schoolreviews.xlsx'
    
    # Download the file as binary using requests
    response = requests.get(github_url)
    
    # Check if the request was successful
    if response.status_code != 200:
        st.error("Error loading the file from GitHub.")
        return None

    # Read the Excel file from the binary content using openpyxl engine
    df = pd.read_excel(BytesIO(response.content), engine='openpyxl')

    # Ensure that all column names are stripped of extra spaces
    df.columns = df.columns.str.strip()

    # Apply sentiment analysis to the 'Reviews' column
    df['Sentiment'] = df['Reviews'].apply(sentiment_analysis)
    
    return df

# Cache the sentiment analysis function
@st.cache_data
def sentiment_analysis(text):
    return TextBlob(text).sentiment.polarity

# Function to generate sentiment distribution graph
def sentiment_distribution_plot(df_filtered, title):
    fig = px.histogram(df_filtered, x='Sentiment', nbins=20, title=title)
    return fig

# Function to generate word cloud
def word_cloud_plot(df_filtered, title):
    text = ' '.join(df_filtered['Reviews'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# Function to generate line chart for sentiment over time
def sentiment_line_plot(df_filtered, title):
    fig = px.line(df_filtered, x='Year', y='Sentiment', title=title, markers=True)
    return fig

# Function to generate pie chart for sentiment distribution
def sentiment_pie_chart(df_filtered, title):
    sentiment_labels = ['Positive' if x > 0 else 'Negative' for x in df_filtered['Sentiment']]
    df_filtered['Sentiment Label'] = sentiment_labels
    fig = px.pie(df_filtered, names='Sentiment Label', title=title)
    return fig

# Function to display metrics (total reviews, positive, and negative)
def display_metrics(df_filtered):
    total_reviews = df_filtered.shape[0]
    positive_reviews = df_filtered[df_filtered['Sentiment'] > 0].shape[0]
    negative_reviews = total_reviews - positive_reviews

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Reviews", value=total_reviews)
    with col2:
        st.metric(label="Positive Reviews", value=f"{positive_reviews} ({(positive_reviews / total_reviews) * 100:.2f}%)")
    with col3:
        st.metric(label="Negative Reviews", value=f"{negative_reviews} ({(negative_reviews / total_reviews) * 100:.2f}%)")

# Main function for NBS School Reviews Page
def nbs_school_reviews_page(df_filtered):
    st.subheader("NBS School Reviews")
    display_metrics(df_filtered)

    # Show graphs in two columns (two graphs per row)
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        fig1 = sentiment_distribution_plot(df_filtered, "Sentiment Distribution for NBS Reviews")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = word_cloud_plot(df_filtered, "Word Cloud of NBS School Reviews")
        st.pyplot(fig2)

    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        fig3 = sentiment_line_plot(df_filtered, "Sentiment Over Time for School Reviews")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = sentiment_pie_chart(df_filtered, "Sentiment Breakdown for School Reviews")
        st.plotly_chart(fig4, use_container_width=True)

# Main function for Course Reviews Page
def course_reviews_page(df_filtered):
    st.subheader("Course Reviews")
    display_metrics(df_filtered)

    # Show graphs in two columns (two graphs per row)
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        fig1 = sentiment_distribution_plot(df_filtered, "Sentiment Distribution for Course Reviews")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = word_cloud_plot(df_filtered, "Word Cloud of Course Reviews")
        st.pyplot(fig2)

    # Show additional graphs (line chart and pie chart)
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        fig3 = sentiment_line_plot(df_filtered, "Sentiment Over Time for Course Reviews")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = sentiment_pie_chart(df_filtered, "Sentiment Breakdown for Course Reviews")
        st.plotly_chart(fig4, use_container_width=True)

# Main function for Action Plan Page
def action_plan_page(df_filtered):
    st.subheader("Action Plan")
    if st.button("Generate Action Plan for Dean"):
        reviews = df_filtered['Reviews'].tolist()
        with st.spinner("Generating action plan..."):
            action_plan = generate_action_plan(reviews)
        st.subheader("Action Plan for Dean")
        st.write(action_plan)

# Function for Raw Reviews Page
def raw_reviews_page(df_filtered):
    st.subheader("Raw Reviews")
    st.dataframe(df_filtered[['Year', 'Month', 'School', 'Course', 'Reviews']])

# Cache OpenAI action plan generation to avoid redundant calls
@st.cache_data
def generate_action_plan(reviews):
    openai.api_key = 'your-api-key-here'  # Replace with your actual API key
    prompt = f"Based on these course reviews: {reviews}\n\nGenerate an actionable plan for the school dean to improve the courses:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=200
    )
    return response.choices[0].text.strip()

# Main function for the Streamlit dashboard
def enhanced_dashboard():
    st.title("ThreadZero - School and Course Reviews")

    # Load data
    df = load_data()

    if df is None:
        st.error("Failed to load data.")
        return

    # Sidebar navigation
    navigation = st.sidebar.selectbox("Select a Page", ["NBS School Reviews", "Course Reviews", "Action Plan", "Raw Reviews"])

    # Filters for school, course, year, and month
    schools = df['School'].unique()
    selected_school = st.sidebar.selectbox("Select School", ['All'] + list(schools))

    courses = df['Course'].unique()
    selected_course = st.sidebar.selectbox("Select Course", ['All'] + list(courses))

    years = ['All'] + list(df['Year'].unique())
    selected_year = st.sidebar.selectbox("Select Year", years)

    # **Check if the 'Month' column exists** before applying the filter
    if 'Month' in df.columns:
        months = ['All'] + list(df['Month'].unique())
        selected_month = st.sidebar.selectbox("Select Month", months)
    else:
        selected_month = 'All'

    sentiments = ['All', 'Positive', 'Negative']
    selected_sentiment = st.sidebar.selectbox("Select Sentiment", sentiments)

    # Apply filters
    df_filtered = df.copy()  # Start with a copy of the full data
    if selected_school != 'All':
        df_filtered = df_filtered[df_filtered['School'] == selected_school]

    if selected_course != 'All':
        df_filtered = df_filtered[df_filtered['Course'] == selected_course]

    if selected_year != 'All':
        df_filtered = df_filtered[df_filtered['Year'] == selected_year]

    if selected_month != 'All' and 'Month' in df.columns:
        df_filtered = df_filtered[df_filtered['Month'] == selected_month]

    if selected_sentiment != 'All':
        sentiment_condition = df_filtered['Sentiment'] > 0 if selected_sentiment == 'Positive' else df_filtered['Sentiment'] <= 0
        df_filtered = df_filtered[sentiment_condition]

    if navigation == "NBS School Reviews":
        nbs_school_reviews_page(df_filtered)
    elif navigation == "Course Reviews":
        course_reviews_page(df_filtered)
    elif navigation == "Action Plan":
        action_plan_page(df_filtered)
    elif navigation == "Raw Reviews":
        raw_reviews_page(df_filtered)

# Run the dashboard
if __name__ == "__main__":
    enhanced_dashboard()
