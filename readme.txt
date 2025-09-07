Wit Access Token: "RDNAC35TZ5OV4VLUR43ACFW4CYPW65UP"
Wit login credential:
    Email: netflixmoviechatbot@gmail.com
    Password: netflixchatbottarumt12345


## Steps to login wit:

1. Visit Wit.ai url: https://wit.ai/
2. Continue with Meta
3. Login with email
4. Login with email password above
5. Go in to app name: netflix_movies_recommender_chatbot
6. In the management tab can check for configuration details: Intents, Entities, Utterances(Training data)


## Step to run the streamlit app (in case the streamlit go inactive):

1. To install the dependencies insert this command in terminal "pip install streamlit pandas requests"
2. Insert this command in terminal to run "streamlit run app.py"
3. Streamlit app will start running in the browser
Here are some samples input(refer the text in wit_samples.json for more):
    show me comedy movies from 2010 directed by edgar wright
    top 10 drama movies
    suggest me any romance movies from 2023


## Evaluation logic:
1. Refer to the test.py
2. Test data is in test_data.json
3. Insert this command in terminal to run "python test.py --json test_data.json --out wit_failures.csv"
4. Results will show in terminal and saves in wit_failures.csv


## Insert entities, intents, utterances logic:

1. Refer to wit_bulk_import.py for logic
2. entities: wit_entities.json
2. intents: wit_intents.json
2. utterances: wit_samples.json

## Data preprocessing logic:

1. Refer to clean_data.py
2. Original movies dataset: netflix_movies_detailed_up_to_2025.csv
3. Cleaned dataset: cleaned_movies.csv