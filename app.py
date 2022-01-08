from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)
model = pickle.load(open('cls.pkl', 'rb'))
@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    d1 = request.form['a']
    d1 = np.asarray(d1, dtype='float64')
    arr = [[d1]]
    pred = model.predict(arr)
    return render_template('result.html', data = pred) 


if __name__ == '__main__':
    app.run(port=5001,debug=True)