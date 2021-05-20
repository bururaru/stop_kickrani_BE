from django.db import models

# Create your models here.
class Kickrani(models.Model):
    kickId = models.AutoField(primary_key=True)
    brand = models.CharField(max_length=250)
    violation = models.IntegerField(default=0, null=True)
    violationImg = models.CharField(max_length=250, null=True)
    kickDate = models.DateTimeField(auto_now=True)