from django.shortcuts import render
from django.http import HttpResponse
from .models import Userprofile
# Create your views here.

def index(request):
#    return HttpResponse("HHS")
    context=Userprofile.objects.all()[1]
    return render(request,'user/index.html',{'context':context})
    