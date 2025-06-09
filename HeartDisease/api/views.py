from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
from .models import load_model_and_scaler, calculate_severity
from django.http import HttpResponse
@csrf_exempt
def predict_heart_disease(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            features = np.array([
                float(data['Age']),
                float(data['Sex']),  # 1: Male, 0: Female
                float(data['ChestPainType']),  # 0: ATA, 1: NAP, 2: ASY, 3: TA
                float(data['RestingBP']),
                float(data['Cholesterol']),
                float(data['FastingBS']),
                float(data['RestingECG']),  # 1: Normal, 2: ST, 0: LVH
                float(data['MaxHR']),
                float(data['ExerciseAngina']),  # 0: No, 1: Yes
                float(data['Oldpeak']),
                float(data['ST_Slope'])  # 2: Up, 1: Flat, 0: Down
            ]).reshape(1, -1)
            
            model, scaler = load_model_and_scaler()
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)
            probability = model.predict_proba(features_scaled).tolist()
            severity = calculate_severity(float(data['Oldpeak']))
            
            return JsonResponse({
                'prediction': int(prediction[0]),  # 0: No disease, 1: Disease
                'probability': probability[0],  # [P(No disease), P(Disease)]
                'severity': severity  # Severity score (0-100)
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)
def home(request):
    return HttpResponse("Welcome to api") 



