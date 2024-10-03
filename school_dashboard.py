import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
import openai

def load_data():
    # Load your Excel file
    df = pd.read_excel(r'C:/Users/Justinian Santos/OneDrive - Nanyang Technological University/Documents/Python Scripts/schoolreviews.xlsx')
    
    # Apply sentiment analysis to the 'Reviews' column
    df['Sentiment'] = df['Reviews'].apply(sentiment_analysis)
    return df

def sentiment_analysis(text):
    return TextBlob(text).sentiment.polarity

def generate_action_plan(reviews):
    openai.api_key = 'your-api-key-here'  # Replace with your actual API key
    prompt = f"Based on these course reviews: {reviews}\n\nGenerate an actionable plan for the school dean to improve the courses:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=200
    )
    return response.choices[0].text.strip()


def display_reviews(filtered_df):
    st.subheader("Filtered Reviews")

    # Ensure 'Course', 'Reviews', 'Sentiment', and 'Year' columns exist
    if not all(col in filtered_df.columns for col in ['Course', 'Reviews', 'Sentiment', 'Year']):
        st.error("One or more required columns ('Course', 'Reviews', 'Sentiment', 'Year') are missing.")
        return

    # Display reviews in an Instagram-style layout
    for index, row in filtered_df.iterrows():
        with st.container():
            st.markdown(f"<div style='background-color: #f9f9f9; padding: 10px; border-radius: 5px;'>", unsafe_allow_html=True)
            st.markdown(f"#### Year: {row['Year']}", unsafe_allow_html=True)
            st.markdown(f"### {row['Course']}", unsafe_allow_html=True)
            st.markdown(f"**Review:** {row['Reviews']}", unsafe_allow_html=True)
            sentiment_label = 'Positive' if row['Sentiment'] > 0 else 'Negative'
            sentiment_color = "green" if sentiment_label == "Positive" else "red"
            st.markdown(f"<span style='color: {sentiment_color};'>**Sentiment:** {sentiment_label}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("---")  # Add a horizontal line for separation

    st.title("ThreadZero")

    df = load_data()

    # Sidebar filters
    years = df['Year'].unique()
    selected_year = st.sidebar.selectbox("Select Year", years)

    sentiments = ['All', 'Positive', 'Negative']
    selected_sentiment = st.sidebar.selectbox("Select Sentiment", sentiments)

    # Add Course Filter
    courses = df['Course'].unique()
    selected_course = st.sidebar.selectbox("Select Course", ['All'] + list(courses))  # Include 'All' option

    # Filter data
    filtered_df = df[df['Year'] == selected_year]
    if selected_course != 'All':
        filtered_df = filtered_df[filtered_df['Course'] == selected_course]
    if selected_sentiment != 'All':
        sentiment_condition = filtered_df['Sentiment'] > 0 if selected_sentiment == 'Positive' else filtered_df['Sentiment'] <= 0
        filtered_df = filtered_df[sentiment_condition]

    # Display the reviews
    display_reviews(filtered_df)

    # Sentiment distribution plot
    fig = px.histogram(filtered_df, x='Sentiment', nbins=20, title='Sentiment Distribution')
    st.plotly_chart(fig)


    # Course rating plot
    if 'Rating' in filtered_df.columns:
        avg_ratings = filtered_df.groupby('Course')['Rating'].mean().sort_values(ascending=False)
        fig = px.bar(avg_ratings, x=avg_ratings.index, y=avg_ratings.values, title='Average Course Ratings')
        st.plotly_chart(fig)

    # Generate action plan
    if st.button("Generate Action Plan for Dean"):
        reviews = filtered_df['Reviews'].tolist()
        action_plan = generate_action_plan(reviews)
        st.subheader("Action Plan for Dean")
        st.write(action_plan)

def enhanced_dashboard():
    st.title("Enhanced School Reviews Dashboard")

    # Load and apply sentiment analysis to the dataset
    df = load_data()

    # Sidebar filters
    years = df['Year'].unique()
    selected_year = st.sidebar.selectbox("Select Year", years)

    sentiments = ['All', 'Positive', 'Negative']
    selected_sentiment = st.sidebar.selectbox("Select Sentiment", sentiments)

    # Add Course Filter
    courses = df['Course'].unique()
    selected_course = st.sidebar.selectbox("Select Course", ['All'] + list(courses))

    # Filter data
    filtered_df = df[df['Year'] == selected_year]
    if selected_course != 'All':
        filtered_df = filtered_df[filtered_df['Course'] == selected_course]
    if selected_sentiment != 'All':
        sentiment_condition = filtered_df['Sentiment'] > 0 if selected_sentiment == 'Positive' else filtered_df['Sentiment'] <= 0
        filtered_df = filtered_df[sentiment_condition]

    # Summary cards
    total_reviews = filtered_df.shape[0]
    positive_reviews = filtered_df[filtered_df['Sentiment'] > 0].shape[0]
    negative_reviews = total_reviews - positive_reviews

    st.subheader(f"Total Reviews: {total_reviews}")
    st.subheader(f"Positive Reviews: {positive_reviews} ({(positive_reviews / total_reviews) * 100:.2f}%)")
    st.subheader(f"Negative Reviews: {negative_reviews} ({(negative_reviews / total_reviews) * 100:.2f}%)")

    # Sentiment distribution plot with explanation
    fig = px.histogram(filtered_df, x='Sentiment', nbins=20, title='Sentiment Distribution',
                       color_discrete_sequence=["#800000", "#000080", "#FFFFFF"])
    st.plotly_chart(fig)

    with st.expander("How was this graph generated?"):
        st.write("""
        **Sentiment Distribution**: 
        This histogram shows the distribution of sentiment scores across reviews. 
        Sentiment is calculated using TextBlob, which assigns a polarity score ranging from -1 (very negative) to 1 (very positive). 
        The x-axis represents these sentiment scores, while the y-axis indicates the count of reviews with a particular sentiment score.
        """)

    # Sentiment by course and year with explanation
    fig = px.bar(filtered_df, x='Course', y='Sentiment', color='Year', title='Sentiment by Course and Year',
                 color_discrete_sequence=["#800000", "#000080", "#FFFFFF"])
    st.plotly_chart(fig)

    with st.expander("How was this graph generated?"):
        st.write("""
        **Sentiment by Course and Year**: 
        This bar chart shows the average sentiment for each course in the selected year. 
        Sentiment scores are aggregated to give an overall measure of how positively or negatively students feel about each course. 
        The x-axis represents the courses, while the y-axis shows the average sentiment score for the course.
        """)

    # Generate action plan
    if st.button("Generate Action Plan for Dean"):
        reviews = filtered_df['Reviews'].tolist()
        action_plan = generate_action_plan(reviews)
        st.subheader("Action Plan for Dean")
        st.write(action_plan)