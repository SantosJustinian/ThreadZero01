import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO


USER_CREDENTIALS = {'admin': 'password123'}
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

# Function to handle login
def login_page():
    st.title("Login")

    # Input fields for username and password
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Login button
    if st.button("Login"):
        # Check if username and password are correct
        if USER_CREDENTIALS.get(username) == password:
            st.session_state['logged_in'] = True
            st.success("Login successful!")
        else:
            st.error("Incorrect username or password")


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

def apply_custom_css():
    st.markdown("""
        <style>
        .review-container {
            text-align: center;
            font-family: 'Aptos', sans-serif;
            font-size: 16px;
            margin-bottom: 20px;
        }
        .review-text {
            margin: 0 auto;
            padding: 10px;
            border-bottom: 1px solid #ddd;
            max-width: 800px;
            
        }
        </style>
    """, unsafe_allow_html=True)

def ntu_reviews_page(df_filtered):
    st.subheader("NTU Reviews")
    display_metrics(df_filtered)

    # Apply custom CSS
    apply_custom_css()

    # Show graphs in two columns (two graphs per row)
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        fig1 = sentiment_distribution_plot(df_filtered, "Sentiment Distribution for NTU Reviews")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = word_cloud_plot(df_filtered, "Word Cloud of NTU Reviews")
        st.pyplot(fig2)

    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        fig3 = sentiment_line_plot(df_filtered, "Sentiment Over Time for NTU Reviews")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = sentiment_pie_chart(df_filtered, "Sentiment Breakdown for NTU Reviews")
        st.plotly_chart(fig4, use_container_width=True)

    # Sort reviews by most recent (Year and Month) and display them
    st.subheader("Recent NTU Reviews")
    df_sorted = df_filtered.sort_values(by=['Year', 'Month'], ascending=[False, False])
    
    for index, row in df_sorted.iterrows():
        st.markdown(f"""
        <div class="review-container">
            <div class="review-text">
                <strong>{row['Year']} {row['Month']}</strong> - {row['Course']}<br>
                {row['Reviews']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Function for NUS Reviews Page
# Main function for NUS Reviews Page with line break between date and review text
def nus_reviews_page(df_filtered):
    st.subheader("NUS Reviews")
    display_metrics(df_filtered)

    # Apply custom CSS
    apply_custom_css()

    # Show graphs in two columns (two graphs per row)
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        fig1 = sentiment_distribution_plot(df_filtered, "Sentiment Distribution for NUS Reviews")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = word_cloud_plot(df_filtered, "Word Cloud of NUS Reviews")
        st.pyplot(fig2)

    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        fig3 = sentiment_line_plot(df_filtered, "Sentiment Over Time for NUS Reviews")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = sentiment_pie_chart(df_filtered, "Sentiment Breakdown for NUS Reviews")
        st.plotly_chart(fig4, use_container_width=True)

    # Sort reviews by most recent (Year and Month) and display them
    st.subheader("Recent NUS Reviews")
    df_sorted = df_filtered.sort_values(by=['Year', 'Month'], ascending=[False, False])
    
    for index, row in df_sorted.iterrows():
        st.markdown(f"""
        <div class="review-container">
            <div class="review-text">
                <strong>{row['Year']} {row['Month']}</strong> - {row['Course']}<br>
                {row['Reviews']}
            </div>
        </div>
        """, unsafe_allow_html=True)



def comparison_page(df_ntu, df_nus):
    st.subheader("Comparison of NTU and NUS Reviews")

    # Calculate total, positive, and negative reviews for each school
    ntu_total = df_ntu.shape[0]
    ntu_positive = df_ntu[df_ntu['Sentiment'] > 0].shape[0]
    ntu_negative = ntu_total - ntu_positive

    nus_total = df_nus.shape[0]
    nus_positive = df_nus[df_nus['Sentiment'] > 0].shape[0]
    nus_negative = nus_total - nus_positive

    # Combine NTU and NUS data for plotting
    comparison_df = pd.DataFrame({
        'School': ['NTU', 'NUS'],
        'Positive Reviews': [ntu_positive, nus_positive],
        'Negative Reviews': [ntu_negative, nus_negative],
        'Total Reviews': [ntu_total, nus_total]
    })

    # Full-width layout for better visibility
    st.subheader("Summary of Reviews")

    # Display total reviews, positive reviews, and negative reviews side by side
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total Reviews (NTU)", value=ntu_total)
        st.metric(label="Total Positive Reviews (NTU)", value=ntu_positive)
        st.metric(label="Total Negative Reviews (NTU)", value=ntu_negative)
    with col2:
        st.metric(label="Total Reviews (NUS)", value=nus_total)
        st.metric(label="Total Positive Reviews (NUS)", value=nus_positive)
        st.metric(label="Total Negative Reviews (NUS)", value=nus_negative)

    # Bar graph comparing positive and negative reviews
    st.subheader("Bar Graph Comparison of Reviews")
    fig = px.bar(
        comparison_df,
        x='School',
        y=['Positive Reviews', 'Negative Reviews'],
        title="Comparison of Positive and Negative Reviews",
        barmode='group',
        labels={'value': 'Number of Reviews'},
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Time-based analysis (Total Reviews and Positive Reviews over time - monthly)
    st.subheader("Monthly Review Trends for NTU and NUS")

    # Group by Year-Month and count total and positive reviews for each month
    df_ntu['Year-Month'] = df_ntu['Year'].astype(str) + '-' + df_ntu['Month']
    df_nus['Year-Month'] = df_nus['Year'].astype(str) + '-' + df_nus['Month']

    # NTU total reviews and positive reviews over time
    ntu_monthly = df_ntu.groupby('Year-Month').size().reset_index(name='Total Reviews')
    ntu_positive_monthly = df_ntu[df_ntu['Sentiment'] > 0].groupby('Year-Month').size().reset_index(name='Positive Reviews')

    # NUS total reviews and positive reviews over time
    nus_monthly = df_nus.groupby('Year-Month').size().reset_index(name='Total Reviews')
    nus_positive_monthly = df_nus[df_nus['Sentiment'] > 0].groupby('Year-Month').size().reset_index(name='Positive Reviews')

    # Merge NTU and NUS data for Total Reviews
    total_reviews_df = pd.merge(ntu_monthly, nus_monthly, on='Year-Month', suffixes=('_NTU', '_NUS'))

    # Line chart for total reviews over time
    st.subheader("Total Reviews Over Time")
    fig_total_reviews = px.line(
        total_reviews_df,
        x='Year-Month', 
        y=['Total Reviews_NTU', 'Total Reviews_NUS'],
        labels={'variable': 'School', 'value': 'Total Reviews'},
        title="Total Reviews Over Time (NTU vs NUS)",
        markers=True
    )
    st.plotly_chart(fig_total_reviews, use_container_width=True)

    # Merge NTU and NUS data for Positive Reviews
    positive_reviews_df = pd.merge(ntu_positive_monthly, nus_positive_monthly, on='Year-Month', suffixes=('_NTU', '_NUS'))

    # Line chart for positive reviews over time
    st.subheader("Positive Reviews Over Time")
    fig_positive_reviews = px.line(
        positive_reviews_df,
        x='Year-Month', 
        y=['Positive Reviews_NTU', 'Positive Reviews_NUS'],
        labels={'variable': 'School', 'value': 'Positive Reviews'},
        title="Positive Reviews Over Time (NTU vs NUS)",
        markers=True
    )
    st.plotly_chart(fig_positive_reviews, use_container_width=True)

    # Add Action Plan
    st.subheader("Generate Action Plan for Both Schools")
    if st.button("Generate Action Plan"):
        reviews_ntu = df_ntu['Reviews'].tolist()
        reviews_nus = df_nus['Reviews'].tolist()
        combined_reviews = reviews_ntu + reviews_nus

        with st.spinner("Generating action plan..."):
            action_plan = generate_action_plan(combined_reviews)
        st.subheader("Action Plan for Dean")
        st.write(action_plan)



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
    # Check if the user is logged in
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    # If the user is not logged in, show the login page
    if not st.session_state['logged_in']:
        login_page()
    else:
        # Show the main content if logged in
        st.title("ThreadZero - School and Course Reviews")

        # Load data
        df = load_data()

        if df is None:
            st.error("Failed to load data.")
            return

        # Sidebar navigation
        # Sidebar navigation for NTU, NUS, and Comparison pages
        navigation = st.sidebar.selectbox("Select a Page", ["NTU Reviews", "NUS Reviews", "Comparison and Action Plan"])

        # Filters for year, month, and sentiment (applies to both NTU and NUS)
        years = ['All'] + list(df['Year'].unique())
        selected_year = st.sidebar.selectbox("Select Year", years)

        if 'Month' in df.columns:
            months = ['All'] + list(df['Month'].unique())
            selected_month = st.sidebar.selectbox("Select Month", months)
        else:
            selected_month = 'All'

        sentiments = ['All', 'Positive', 'Negative']
        selected_sentiment = st.sidebar.selectbox("Select Sentiment", sentiments)

        # Apply filters for NTU and NUS data separately
        df_ntu = df[df['School'] == 'NTU']
        df_nus = df[df['School'] == 'NUS']

        if selected_year != 'All':
            df_ntu = df_ntu[df_ntu['Year'] == selected_year]
            df_nus = df_nus[df_nus['Year'] == selected_year]

        if selected_month != 'All' and 'Month' in df.columns:
            df_ntu = df_ntu[df_ntu['Month'] == selected_month]
            df_nus = df_nus[df_nus['Month'] == selected_month]

        if selected_sentiment != 'All':
            sentiment_condition_ntu = df_ntu['Sentiment'] > 0 if selected_sentiment == 'Positive' else df_ntu['Sentiment'] <= 0
            df_ntu = df_ntu[sentiment_condition_ntu]

            sentiment_condition_nus = df_nus['Sentiment'] > 0 if selected_sentiment == 'Positive' else df_nus['Sentiment'] <= 0
            df_nus = df_nus[sentiment_condition_nus]


        # Apply filters
        df_filtered = df.copy()  # Start with a copy of the full data
        # Main navigation logic for NTU, NUS, and Comparison pages
        if navigation == "NTU Reviews":
            ntu_reviews_page(df_ntu)
        elif navigation == "NUS Reviews":
            nus_reviews_page(df_nus)
        elif navigation == "Comparison and Action Plan":
            comparison_page(df_ntu, df_nus)

    # Run the dashboard
if __name__ == "__main__":
    enhanced_dashboard()
