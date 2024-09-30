from flask import Flask, render_template, request
import pickle
import numpy as np

# Load models
Decision_Tree_Model = pickle.load(open('DecisionTree_Model.pkl', 'rb'))
Random_Forest_Model = pickle.load(open('RandomForest_Model.pkl', 'rb'))
AdaBoost_Model = pickle.load(open('AdaBoost_Model.pkl', 'rb'))
XGBoost_Model = pickle.load(open('XGBoost_Model.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # HTML page to display the form

@app.route('/predict', methods=['POST'])
def predict():
    model_option = request.form['model_option']  # Get model option from form

    # Get features from the form
    Feature1 = float(request.form['latitude'])
    Feature2 = float(request.form['longitude'])
    Feature3 = float(request.form['stars'])
    Feature4 = float(request.form['review_count_x'])
    Feature5 = float(request.form['avg_review_rating'])
    Feature6 = float(request.form['review_count_y'])
    Feature7 = float(request.form['checkin_count'])
    Feature8 = float(request.form['tip_count'])

    # Create the feature array (assuming there are no Feature9 and Feature10)
    features = np.array([[Feature1, Feature2, Feature3, Feature4, Feature5, Feature6, Feature7,Feature8]])

    # Model selection
    if model_option == 'Decision Tree':
        model = Decision_Tree_Model
    elif model_option == 'Random Forest':
        model = Random_Forest_Model
    elif model_option == 'AdaBoost':
        model = AdaBoost_Model
    elif model_option == 'XGBoost':
        model = XGBoost_Model
    else:
        return render_template('result.html', prediction_text='Invalid model selected.')

    # Prediction
    prediction = model.predict(features)
    if prediction[0] == 1:
        result = 'Business is Open.'
    else:
        result = 'Business is not Open.'

    return render_template('result.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
