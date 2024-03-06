from django.shortcuts import render
from MachineLearning.Logistic_Regression import Ammar

    
def home(request):
    if request.method=="POST":
        age=request.POST.get('age')
        new=Ammar.logistic(age)
        return render(request,"home.html",{"lis":new})
    else:
        new=['Analyze']
        return render(request,"home.html",{"lis":new})
    

    