from flask import Flask, request, Response
from flask import render_template
from Predictor import *
import os
dirname = os.path.dirname(__file__)
os.chdir(dirname)

app = Flask(__name__)

pred = Predictor('data/mining.csv')
pred.preprocess()
pred.load_models()


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/query", methods=["GET", "POST"])
def query():
    ans = pred.predict(request.form.to_dict())
    return render_template("results.html", **ans)


app.run()
