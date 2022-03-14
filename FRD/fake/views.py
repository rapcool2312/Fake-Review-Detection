from django.shortcuts import render
from django.http import HttpResponse
import os
import joblib

model1 = joblib.load(os.path.dirname(__file__) + "\\myModel2.pkl")
model2 = joblib.load(os.path.dirname(__file__) + "\\mySVCModel1.pkl")

# Create your views here.
def index(request):
    return render(request, 'index.html')

def checkfake(request):
    print(">>>>>>>>>>>>>>>>>>" + request.method)
    if(request.method == "POST"):
        finalAns = ""
        algo = request.POST.get("algo")
        rawData = request.POST.get("rawdata")

        if(algo == "Algo-1"):
            finalAns = model1.predict([rawData])[0]
            param = {"answer" : finalAns}
        elif(algo == "Algo-2"):
            finalAns = model2.predict([rawData])[0]
            param = {"answer" : finalAns}


        # print(algo)
        # print(rawData)

        print(finalAns)



        return render(request, 'output.html', param)
    else:
        return render(request, 'index.html')
