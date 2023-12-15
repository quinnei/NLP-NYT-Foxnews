import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import string


### Set the page layout
st.set_page_config(layout = "wide", page_title = "Climate Change Discourse in U.S. Media")


### Read in the data

def load_data(file_name):
    base_url = 'https://raw.githubusercontent.com/quinnei/NLP-NYT-Foxnews/main/Data/'
    file_path = f"{base_url}{file_name}.csv"
    data = pd.read_csv(file_path) 
    data['date'] = pd.to_datetime(data['date'])
    return data


NYT = load_data('NYT')
foxnews = load_data('Foxnews')


### Add & remove stopwords

add_to_stopwords = {"climate", "change", "global", "warming", "warm", "environment", "environmental", "planet", "earth",
                    "study", "studies", "research", "report", "reporter", "analysis", "data", "datum", "according",
                    "work", "project", "plan", "program", "researcher", "expert", "scientist", "record", "source",
                    "people", "public", "thing", "community", "group", "region", "role", "method", "way", "road", "path",
                    "says", "said", "announce", "suggest", "suggests", "show", "shows", "explain", "explains", "explained", "release", "reveal", "find", "finds",
                    "know", "question", "answer", "think", "include", "mean", "means",
                    "propose", "expect", "reach", "come", "bring", "start", "create", "run",
                    "help", "continue", "want", "need", "require",
                    "like", "likely", "look", "looks", "nearly", "near", "away",
                    "fight", "combat", "address", "face", "action", "effort", "try", "challenge",
                    "impact", "effect", "affect", "meet", "reach", "goal", "aim", "objective",
                    "year", "month", "month", "week", "weeks", "days", "decade", "century", "ago", "time", "late", "early", "future", "past", "yesterday", "today", "tomorrow", "recent",
                    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                    "hundred", "thousand", "million", "billion", "percent", "half", "quarter", "number",
                    "average", "common"}

# Combine default stopwords and my own. Convert to a list
my_stopwords = ENGLISH_STOP_WORDS.union(add_to_stopwords)
my_stopwords_list = list(my_stopwords)



### Search for the top ngrams, among articles that were published in a given year and month
### -> based on document frequency (the number of articles that contained a given word)

def get_top_ngrams(data, year, month, top_n = 15):
    
    # Filter the articles based on year and month
    filtered_data = data[(data['date'].dt.year == year) & (data['date'].dt.month == month)]
    
    # Clean the abstracts of the articles
    def preprocess_text(text):
        
        # Convert to lowercase
        text = text.lower()
        
        # Define strings that I want to remove - punctuations and digits
        # This is to keep and to clean words such as 'COP26', 'COP28' -> 'COP' / 'E.P.A' -> 'EPA'
        translation_table = str.maketrans('', '', string.punctuation + string.digits)

        # Apply the translation table to clean the abstract
        text = text.translate(translation_table)
        
        return text


    cleaned_abstracts = filtered_data['abstract'].apply(preprocess_text)

    
    # Initialize the CountVectorizer:
    vectorizer = CountVectorizer(stop_words = my_stopwords_list, # with custom stopwords
                                 ngram_range = (1, 3), # with not only unigrams but also bigrams and trigrams
                                 binary = True) # for calculating *document* frequency

    # Apply the countvectorizer to the cleaned abstracts
    X = vectorizer.fit_transform(cleaned_abstracts)

    # Count how many documents contained a given word (i.e. document frequency)
    # Convert the matrix to a 1D array, to add as column
    document_frequency = np.ravel(X.sum(axis = 0))
    words = vectorizer.get_feature_names_out()
    ngrams = pd.DataFrame({
        "Words": words,
        "Document Frequency": document_frequency})

    # List the ngrams within articles published in a given year & month 
    # The most frequency occurring across documents = top
    top_ngrams = ngrams.sort_values(by = "Document Frequency", ascending = False).head(top_n)
    
    return top_ngrams.reset_index(drop = True)





### Define a function that plots a barchart of the top 15 most frequent words
def plot_top_ngrams(data, year, month, source):
    
    # Construct a dataframe that stores most frequently occurring terms across articles
    top_ngrams = get_top_ngrams(data, year, month) 
    
    fig, ax = plt.subplots(figsize = (6, 4))
    
    # Specify the colors of the bars: for Left-leaning news outlet -> blue; for Right-leaning -> red
    color = 'royalblue' if source == "NYT" else 'orangered'

    # Plot a horizontal bar plot
    ax.barh(top_ngrams['Words'], # y-axis
            top_ngrams['Document Frequency'], # x-axis
            color = color)

    # Flip the y-axis so that the highest frequency comes at the top
    ax.invert_yaxis()


    ax.set_title(f'Most frequent words in {year}-{month}', fontweight = 'bold')
    ax.set_xlabel('\nNumber of articles', fontweight = 'bold')
    ax.set_ylabel('Words', fontweight = 'bold')
    # Set the x-axis to display only integer values
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer = True))
    
    # Make the top 7 words boldface
    for i, label in enumerate(ax.get_yticklabels()):
        if i < 7:  # Apply bold fontweight to the top 7 labels
            label.set_fontweight('bold')


    # Add dashed lines for the grid.
    plt.grid(linestyle = '--')

    st.pyplot(fig)
    
    
    
    
 ### Design the app ### 
     
def main():
    st.title("The 15 Most Frequent Words in Climate Related News Articles")

    # Combine the datasets to find all unique year-month combinations
    all_dates = pd.concat([NYT['date'], foxnews['date']]).dt.to_period('M').unique()
    all_dates = all_dates.strftime('%Y-%m')
    
    selected_date = st.selectbox(
        'Select Year-Month:', 
        options = sorted(all_dates, reverse = True)
    )

    year, month = map(int, selected_date.split('-'))

    # Create two columns for the plots
    col1, col2 = st.columns(2)

    # Plotting for NYT in the first column
    with col1:
        st.subheader("New York Times")
        plot_top_ngrams(NYT, year, month, "NYT")

    # Plotting for Fox News in the second column
    with col2:
        st.subheader("Fox News")
        plot_top_ngrams(foxnews, year, month, "Fox News")


if __name__ == "__main__":
    main()