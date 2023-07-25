
# 1. Library imports
import uvicorn
from fastapi import FastAPI
import pandas as pd
import tensorflow as tf
import keras
#from typing_extensions import Required, NotRequired, TypeAliasType
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from mangum import Mangum
# 2. Create the app object
app = FastAPI()
handler = Mangum(app)

classifier=keras.models.load_model("my_model.keras")
df = pd.read_csv('https://raw.githubusercontent.com/kenneth-lee-ch/SMS-Spam-Classification/master/spam.csv', encoding='ISO-8859-1')
# rename the columns
df = df[['v1','v2']]
df.rename(columns={'v1':'label', 'v2':'message'}, inplace=True)
df_majority = df[(df['label']=='ham')] 
df_minority = df[(df['label']=='spam')] 
df_majority = df_majority.sample(len(df_minority), random_state=0)
msg_df = pd.concat([df_majority,df_minority])
msg_df = msg_df.sample(frac=1, random_state=0)
msg_df['text_length'] = msg_df['message'].apply(len)
msg_df['msg_type'] = msg_df['label'].map({'ham':0, 'spam':1})
msg_label = msg_df['msg_type'].values
x_train, x_test, y_train, y_test = train_test_split(msg_df['message'], msg_label, test_size=0.2, random_state=434)


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Smishy': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predictor(predict_msg):
    tokenizer = Tokenizer(num_words = 500,  char_level = False, oov_token = '<OOV>')
    tokenizer.fit_on_texts(x_train)
    seq = tokenizer.texts_to_sequences([predict_msg])
    padded = pad_sequences(seq, maxlen=50)
    predictionperc = int ((classifier.predict(padded))*100)
    prediction = "According to ML model, this text is " + str(predictionperc) + "% spam"
    return {
        'prediction': prediction
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload