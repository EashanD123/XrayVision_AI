from flask import Flask, redirect, url_for, render_template, request

app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('home.html')

@app.route('/possible_diseases')
def possible_diseases():
    return render_template('possible_diseases.html')

@app.route('/test_disease')
def test_disease():
    return render_template('test_disease.html')

if __name__ == '__main__':
    app.run(debug=True)
