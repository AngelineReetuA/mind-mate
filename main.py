from flask import Flask, request, render_template
from chat import getResponse
app = Flask(__name__)

@app.route("/")
def hello_world():
    answer = ""
    return render_template('index.html')

@app.route("/question")
def getAnswer():
    question = request.args.get('q')
    answer = getResponse(question)
    return answer

