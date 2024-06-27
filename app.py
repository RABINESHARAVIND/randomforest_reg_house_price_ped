from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
min_max_scaler = pickle.load(open('min_max_scaler.pkl', 'rb'))
standard_scaler = pickle.load(open('standard_scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            val1 = request.form['bedrooms']
            val2 = request.form['bathrooms']
            val3 = request.form['sqft_living']
            val4 = request.form['sqft_lot']
            val5 = request.form['floors']
            val6 = request.form['waterfront']
            val7 = request.form['view']
            val8 = request.form['condition']
            val9 = request.form['grade']
            val10 = request.form['sqft_above']
            val11 = request.form['sqft_basement']
            val12 = request.form['yr_built']

            if not all([val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12]):
                return render_template('index.html', error="Please provide all input values.")

            try:
                data = {
                    'bedrooms': [float(val1)],
                    'bathrooms': [float(val2)],
                    'sqft_living': [float(val3)],
                    'sqft_lot': [float(val4)],
                    'floors': [float(val5)],
                    'waterfront': [float(val6)],
                    'view': [float(val7)],
                    'condition': [float(val8)],
                    'grade': [float(val9)],
                    'sqft_above': [float(val10)],
                    'sqft_basement': [float(val11)],
                    'yr_built': [float(val12)]
                }
                df = pd.DataFrame(data)
            except ValueError:
                return render_template('index.html', error="Invalid input. Please enter numeric values.")

            df_normalized = min_max_scaler.transform(df)

            df_standardized = standard_scaler.transform(df_normalized)

            pred = model.predict(df_standardized)

            return render_template('index.html', data=int(pred))
    except Exception as e:
        print(f"Exception occurred: {e}")
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
