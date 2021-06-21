from django.db import models

class Image(models.Model):
    image_ID = models.CharField(primary_key=True,max_length=250)
    datetime = models.CharField(max_length=250, null=True)
    location = models.CharField(max_length=250, null=True)
    rider_number=models.IntegerField(default=0, null=True)


# Create your models here.
class Rider(models.Model):
    rider_ID = models.AutoField(primary_key=True)
    image_ID = models.ForeignKey(Image, on_delete=models.CASCADE)
    rider_location = models.CharField(max_length=250, null=True)
    rider_percentage= models.FloatField(max_length=250, null=True)
    brand = models.CharField(max_length=250, null=True)
    helmet_number = models.IntegerField(default=0, null=True)
    person_number = models.IntegerField(default=0, null=True)
    foul= models.IntegerField(default=0, null=True) # 1: 2인이상 탑승 위반, 2: 헬멧 미착용 위반, 3: 2인이상 및 헬멧 위반
    datetime = models.CharField(max_length=250, null=True)
    location = models.CharField(max_length=250, null=True)
    num_brand = models.IntegerField(default=0, null=True)


class Rider_information(models.Model):
    information_Id=models.AutoField(primary_key=True)
    rider_ID=models.ForeignKey(Rider, on_delete=models.CASCADE)
    helmet_location= models.CharField(max_length=250, null=True)
    helmet_percentage= models.FloatField(max_length=250, null=True)
    person_location= models.CharField(max_length=250, null=True)
    person_percentage= models.CharField(max_length=250, null=True)
