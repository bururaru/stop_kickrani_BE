from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import Kickrani
from .serializer import KickraniSerializer

@api_view(['GET'])
def kickraniList(request):
    kickranis = Kickrani.objects.all()
    serializer = KickraniSerializer(kickranis, many=True)
    return Response(serializer.data)