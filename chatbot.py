import nltk
import string
import random
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

f = open('data', 'r', errors='ignore')
raw_doc = f.read()
# preprocessing
row_doc = raw_doc.lower() #lower case
nltk.download('punkt') #tokenizer
nltk.download('wordnet') # dictionary
nltk.download('omw-1.4')
sentance_tokens = nltk.sent_tokenize(row_doc)
word_tokens = nltk.word_tokenize(row_doc)

# lemmitization
lemmer = nltk.stem.WordNetLemmatizer()
def lemtoken(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punc_dict = dict((ord(punct),None) for punct in string.punctuation)

def lemnormalize(text):
    return lemtoken(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

greet_input = ('hello', 'hi', 'hey','how are you', 'whatssup')
greet_response = ('hi','hey', 'hey, There', 'well look who is here')
def greet(sentance):
    for word in sentance.split():
        if word.lower() in greet_input:
            return random.choice(greet_response)

def response(user_response):
    robo1_response = ''
    TfidVec = TfidfVectorizer(tokenizer=lemnormalize, stop_words='english')
    tfidf = TfidVec.fit_transform(sentance_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo1_response = robo1_response+"I am sorry, Unable to understand you"
        return robo1_response
    else:
        robo1_response = robo1_response+sentance_tokens[idx]
        return robo1_response

flag = True
print("hello! I am the reinforcement leanring bot. Start typing your text after greeting to talks to me. for ending conversation type bye")
while(flag == True):
    user_response = input()
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response == 'thank you' or user_response == 'thanks':
            flag = False
            print("Bot , Your welcome")
        else:
            if(greet(user_response)!= None):
                print('Bot '+greet(user_response))
            else:
                sentance_tokens.append(user_response)
                word_tokens = word_tokens +nltk.word_tokenize(user_response)
                final_word = list(set(word_tokens))
                print('Bot ', end='')
                print(response(user_response))
                sentance_tokens.remove(user_response)
    else:
        flag = False
        print('Bot: Good bye')

