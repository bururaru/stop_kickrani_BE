from django.db import models
# from storages.backends.s3boto3 import S3Boto3Storage
#
# class MediaStorage(S3Boto3Storage):
#     location = 'image'
#     file_overwrite = False

# Create your models here.
class Kickrani(models.Model):
    kickId = models.AutoField(primary_key=True)
    brand = models.CharField(max_length=250, null=True)
    helmet = models.IntegerField(default=0, null=True)
    person = models.IntegerField(default=0, null=True)
    image = models.CharField(max_length=250, null=True)
<<<<<<< HEAD
=======
    #image = models.ImageField(upload_to='image')
>>>>>>> root_main
    datetime = models.CharField(max_length=250, null=True)
    location = models.CharField(max_length=250, null=True)
    num_brand = models.IntegerField(default=0, null=True)
    violation= models.IntegerField(default=0, null=True) # 1: 2인이상 탑승 위반, 2: 헬멧 미착용 위반, 3: 2인이상 및 헬멧 위반