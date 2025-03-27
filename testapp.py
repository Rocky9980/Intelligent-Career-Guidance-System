from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def career():
    return render_template("hometest.html")

@app.route('/predict', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form
        print("Raw form data:", result)
        
        res = result.to_dict(flat=True)
        print("Dictionary format:", res)
        
        arr = list(res.values())
        print("Raw values array:", arr)
        
        try:
            # Convert input data to a NumPy array of float type
            data = np.array(arr, dtype=float).reshape(1, -1)
            print("Processed numerical input:", data)
        except ValueError as e:
            print("Error converting input to numeric:", e)
            return "Invalid input: Please enter valid numeric values."

        # Load the trained model
        loaded_model = pickle.load(open("careerlast.pkl", 'rb'))
        
        # Make predictions
        predictions = loaded_model.predict(data)
        pred_proba = loaded_model.predict_proba(data)
        print("Predicted class:", predictions[0])
        print("Prediction probabilities:", pred_proba)
        
        # Thresholding probabilities
        pred_bool = pred_proba > 0.05
        res = {index: j for index, j in enumerate(range(17)) if pred_bool[0, j]}
        
        final_res = {index: values for index, values in res.items() if values != predictions[0]}
        
        jobs_dict = {
            0: 'AI ML Specialist', 1: 'API Integration Specialist', 2: 'Application Support Engineer',
            3: 'Business Analyst', 4: 'Customer Service Executive', 5: 'Cyber Security Specialist',
            6: 'Data Scientist', 7: 'Database Administrator', 8: 'Graphics Designer',
            9: 'Hardware Engineer', 10: 'Helpdesk Engineer', 11: 'Information Security Specialist',
            12: 'Networking Engineer', 13: 'Project Manager', 14: 'Software Developer',
            15: 'Software Tester', 16: 'Technical Writer'
        }
        
        return render_template("testafter.html", final_res=final_res, jobs_dict=jobs_dict, job0=predictions[0])

if __name__ == '__main__':
    app.run(debug=True)
