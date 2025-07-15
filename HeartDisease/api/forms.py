from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Profile, Feedback

class CustomUserRegisterForm(UserCreationForm):
    age = forms.IntegerField(required=True, label='Age')
    gender = forms.ChoiceField(choices=Profile.GENDER_CHOICES, required=True, label='Gender')

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'age', 'gender', 'password1', 'password2']


class FeedbackForm(forms.ModelForm):
    class Meta:
        model = Feedback
        fields = ['comment', 'rating']
        widgets = {
            'comment': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Enter your feedback here...'
            }),
            'rating': forms.Select(attrs={
                'class': 'form-control'
            }),
        }
        labels = {
            'comment': 'Your Feedback',
            'rating': 'Rating (1-5)',
        }



