from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
import joblib
import numpy as np
#from .models import HeartDiseaseModel  # Assuming model is in models.py

# Existing home view (unchanged)
def home(request):
    return render(request, 'home3.html')

# Modified predict view
def predict(request):
    if request.method == 'POST':
        try:
            # Extract form data
           
         
            Age=request.POST.get('Age')
            Sex= int(request.POST.get('Sex'))
            ChestPainType=int(request.POST.get('ChestPainType'))
            RestingBP= int(request.POST.get('RestingBP'))
            Cholesterol= int(request.POST.get('Cholesterol'))
            FastingBS= int(request.POST.get('FastingBS'))
            RestingECG= int(request.POST.get('RestingECG'))
            MaxHR= int(request.POST.get('MaxHR'))
            ExerciseAngina= int(request.POST.get('ExerciseAngina'))
            Oldpeak= float(request.POST.get('Oldpeak'))
            ST_Slope= int(request.POST.get('ST_Slope'))
            print(Age)
            data = {
                'Age': Age,
                'Sex': Sex,
                'ChestPainType': ChestPainType,
                'RestingBP': RestingBP,
                'Cholesterol': Cholesterol,
                'FastingBS': FastingBS,
                'RestingECG': RestingECG,
                'MaxHR': MaxHR,
                'ExerciseAngina': ExerciseAngina,
                'Oldpeak': Oldpeak,
                'ST_Slope': ST_Slope,
            } 
         
         
            """ data = {
                'Age': int(request.POST['Age']),
                'Sex': int(request.POST['Sex']),
                'ChestPainType': int(request.POST['ChestPainType']),
                'RestingBP': int(request.POST['RestingBP']),
                'Cholesterol': int(request.POST['Cholesterol']),
                'FastingBS': int(request.POST['FastingBS']),
                'RestingECG': int(request.POST['RestingECG']),
                'MaxHR': int(request.POST['MaxHR']),
                'ExerciseAngina': int(request.POST['ExerciseAngina']),
                'Oldpeak': float(request.POST['Oldpeak']),
                'ST_Slope': int(request.POST['ST_Slope']),
               }"""

            # Load the trained model
            model = joblib.load('models/heart_disease_model.joblib')  # Adjust path as needed

            # Prepare input for prediction
            input_data = np.array([list(data.values())]).reshape(1, -1)

            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            severity = min(max(int(probability[1] * 100), 0), 100)  # Scale to 0-100

            # Store in session
            """request.session['prediction'] = prediction
            request.session['probability'] = probability.tolist()
            request.session['severity'] = severity"""
            request.session['prediction'] = int(prediction)
            request.session['probability'] = [float(prob) for prob in probability.tolist()]
            request.session['severity'] = int(severity)


            # Redirect to result page
            return redirect('result')
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            messages.error(request, f"Error: {str(e)}")
            return redirect('home')
    return render(request, 'home3.html')

# New result view
def result(request):
    if 'prediction' not in request.session or 'probability' not in request.session or 'severity' not in request.session:
        return redirect('home')

    prediction = request.session['prediction']
    probability = request.session['probability']
    severity = request.session['severity']

    # Clear session data after use
    del request.session['prediction']
    del request.session['probability']
    del request.session['severity']

    # Determine severity level
    severity_level = 'Low' if severity < 10 else 'Moderate' if severity <= 50 else 'High'

    # Generate advice message
    if prediction == 1:  # Heart Disease predicted
        if severity_level == 'High':
            advice_message = "⚠️ Please consult a cardiologist immediately. Your risk level is high."
        elif severity_level == 'Moderate':
            advice_message = "🩺 It's advised to get a medical check-up soon. Lifestyle changes may help."
        else:  # Low severity
            advice_message = "😊 Mild risk detected. Regular exercise and healthy eating are recommended."
    else:  # No Heart Disease predicted
        advice_message = "🎉 You seem to be healthy! Keep up the good habits and get regular checkups."

    context = {
        'prediction': 'No Heart Disease' if prediction == 0 else 'Heart Disease',
        'probability_no_disease': '{:.2f}'.format(probability[0] * 100),
        'probability_disease': '{:.2f}'.format(probability[1] * 100),
        'severity': severity,
        'severity_level': severity_level,
        'advice_message': advice_message
    }

    return render(request, 'result.html', context)

