from django.db import models

# Create your models here.
class Kickrani(models.Model):
    kickId = models.AutoField(primary_key=True)
    brand = models.CharField(max_length=250)
    helmet = models.IntegerField(default=0, null=True)
    person = models.IntegerField(default=0, null=True)
    image = models.CharField(max_length=250, null=True)
    datetime = models.DateTimeField(auto_now=True)
    location = models.CharField(max_length=250, null=True)
    violation= models.IntegerField(default=0, null=True) # 1: 2인이상 탑승 위반, 2: 헬멧 미착용 위반, 3: 2인이상 및 헬멧 위반