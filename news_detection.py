import re
import pickle
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.ensemble import RandomForestClassifier
from colorama import Fore, init
init()

RED = Fore.RED
CYAN = Fore.CYAN
GREEN = Fore.GREEN
RESET = Fore.RESET
MAGENTA = Fore.MAGENTA
YELLOW = Fore.YELLOW

# Function to clean text
def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r'https?://\S+|www\.\S+', '', txt)
    txt = re.sub(r'<.*?>', '', txt)
    txt = re.sub(r'[^a-zA-Z ]+', '', txt)
    txt = ' '.join(txt.split())
    return txt

def output_label(n):
    return "Fake News!!!" if n == 0 else "Real News!!!"

# Vectorizer initialization
try:

    vectorization = pickle.load(open('vectorizer.sav', 'rb'))
    print(f'{GREEN}Vectorizer Loaded!{RESET}')
    model = pickle.load(open('model.sav', 'rb'))
    print(f'{GREEN}Model Loaded!{RESET}')

except:

    model = RandomForestClassifier()

    # Load and preprocess data
    true = pd.read_csv('data/true.csv')
    fake = pd.read_csv('data/fake.csv')

    true['label'] = 1
    fake['label'] = 0
    news = pd.concat([true, fake], axis=0)
    news = news.drop(['title', 'subject', 'date'], axis=1)
    # Shuffle the dataset
    news = news.sample(frac=1).reset_index(drop=True) 

    # Clean the text in the dataset
    news['text'] = news['text'].apply(clean_text)

    # Split data
    x, y = news['text'], news['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    print(f'{CYAN}Data Split into Train and Test{RESET}')

    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    print(f'{GREEN}Vectorizer Trained!{RESET}')
    pickle.dump(vectorization, open('vectorizer.sav', 'wb'))

    # Transform train and test data
    xv_train = vectorization.transform(x_train)
    xv_test = vectorization.transform(x_test)

    print(f'{CYAN}Training the Model...{RESET}')
    model.fit(xv_train, y_train)
    print(f'{GREEN}Model Trained!{RESET}')
    pickle.dump(model, open('model.sav', 'wb'))

    # Model Evaluation
    accuracy = model.score(xv_test, y_test)
    print(f'{YELLOW}Model Accuracy: {accuracy:.2f}{RESET}')

# Function to get news status (Real or Fake)
def get_news_status(news):
    testing_news = {"text": [news]}
    new_df_test = pd.DataFrame(testing_news)
    new_df_test["text"] = new_df_test["text"].apply(clean_text)
    new_xv_test = vectorization.transform(new_df_test["text"])
    prediction = model.predict(new_xv_test)[0]
    return prediction

if __name__ == "__main__":
    news = '''
US Elections Results: Donald Trump has won the US elections, winning a landslide victory in swing states. His rival Kamala Harris has secured 226 electoral colleges, while Trump has won 295—crossing the majority mark of 270. He will be the second Republican to get a second term in office in 20 years. George Bush, a Republican, was president from 2001 to 2009. 

As far as swing states are concerned, Trump has already won the swing states of Georgia, North Carolina, Pennsylvania, Michigan, Wisconsin, and is leading in the other two - Arizona, and Nevada.

Kamala Harris conceded defeat early morning. In her speech, the Democratic leader said though she has conceded the election, she has not conceded "the fight that fuelled this campaign."
'''.strip()
    
    print(f'{MAGENTA}News: {YELLOW}{news}')
    print(f'{MAGENTA}Status: {CYAN}{"TRUE" if get_news_status(news) else "FAKE"}{RESET}')