from flask import Flask,render_template,request
import random
app = Flask(__name__)
def func1(s):
	# Do something here
	l = ["23","123","134","7342589","342","23479","234"]
	l.sort()
	if s == "" or not (s.isnumeric()):
		return 
	elif s.isnumeric():
		ss = int(s)
		if ss<0 or ss>=len(l):
			return
	return l[int(s)]
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