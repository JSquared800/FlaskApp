from flask import Flask,render_template,request,Response
import requests
import random
import tweepy
app = Flask(__name__)
from transformers import Trainer, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
import time
import numpy as np
import os
import matplotlib
import io
import matplotlib.pyplot as plt
import numpy as np
tokenizer = AutoTokenizer.from_pretrained("FlaskApp/Roberta-Large")
model = AutoModelForSequenceClassification.from_pretrained("FlaskApp/Roberta-Large", num_labels=3)
model.load_state_dict(torch.load("FlaskApp/fold1_modelroberta-large_epoch6.pth", map_location=torch.device('cpu')))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
def func1(s):
	# Do something here
	#l = ["23","123","134","7342589","342","23479","234"]
	#l.sort()
	#if s == "" or not (s.isnumeric()):
	#	return 
	#elif s.isnumeric():
	#	ss = int(s)
	#	if ss<0 or ss>=len(l):
	#		return
	#return l[int(s)]
	if(s == ""):
		return
	client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAAORicgEAAAAAy%2BK6TiiRsjrMVeGz7yTjaM%2B9R%2BM%3DVk5nXSbAU9sJ9Oyd3GnuSdDe30QHSTa0drJNCmTDwq9GP0vfPK')
	query = s + " finance"

	tweets = client.search_recent_tweets(query=query, tweet_fields=['context_annotations', 'created_at'], max_results=49)
	ss = ""
	for tweet in tweets.data:
		ss+=tweet.text
		ss+=' ||| '
	print(ss)
	return PredictTweet(ss.split(" ||| "))
def PredictTweet(tweet_array):
	
	fout = open("FlaskApp/static/model_response.txt","w", encoding="utf-8")
	counts = np.zeros(3)
	result = ""
	for tweet in tweet_array:
		conv_dict = {
			0 : "positive",
            1 : "neutral",
            2 : "negative"
        }
		tokenized_inputs = tokenizer(tweet, max_length=49, truncation=True, padding="max_length", is_split_into_words=False)
		tokenized_inputs["input_ids"] = torch.tensor(tokenized_inputs["input_ids"]).unsqueeze(0)
		tokenized_inputs["attention_mask"] = torch.tensor(tokenized_inputs["attention_mask"]).unsqueeze(0)
		raw_probas = torch.nn.Softmax(dim=-1)(model(tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"]).logits)
		class_pred = torch.argmax(raw_probas, dim=-1).item()
		counts[class_pred] += 1
		result += tweet
		result += f" | Model Classified This Tweet As {conv_dict[class_pred]}\n"
	fout.write(result)
	fout.close()
	return f"There are {int(counts[0])*2}% positive tweets, {int(counts[1])*2}% neutral tweets, and {int(counts[2])*2}% negative tweets"
@app.route("/")
@app.route("/home")
def home():
	return render_template("index.html")
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


@app.route("/result",methods = ['POST','GET'])
def result():
	output = request.form.to_dict()
	name = output["name"]	
	time.sleep(1)
	return render_template("index.html",name = func1(name))

if __name__ == '__main__':
	app.run(debug = True,port = 5001)