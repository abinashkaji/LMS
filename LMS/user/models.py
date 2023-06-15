from django.db import models

# Create your models here.

class Userprofile(models.Model):
    name=models.CharField(max_length=30,default="abc")
    image=models.ImageField()
    date=models.DateField(auto_now=False, auto_now_add=True)
    
    def __str__(self) -> str:
        return f'Hello {self.name} '