from flask import Flask,render_template,request
import requests
import random
import tweepy
app = Flask(__name__)
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
	query = s

	tweets = client.search_recent_tweets(query=query, tweet_fields=['context_annotations', 'created_at'], max_results=100)
	ss = ""
	for tweet in tweets.data:
		ss+=tweet.text
		ss+=' ||| '
		if len(tweet.context_annotations) > 0:
			print(tweet.context_annotations)
	return ss
	
@app.route("/")
@app.route("/home")
def home():
	return render_template("index.html")

@app.route("/result",methods = ['POST','GET'])
def result():
	output = request.form.to_dict()
	name = output["name"]

	return render_template("index.html",name = func1(name))

if __name__ == '__main__':
	app.run(debug = True,port = 5001)