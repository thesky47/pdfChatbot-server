from rest_framework import generics
from rest_framework import status
from rest_framework.response import Response
from .models import CustomUser
from .serializers import UserRegistrationSerializer
from rest_framework_simplejwt.views import TokenObtainPairView

class UserRegistrationView(generics.CreateAPIView):
    queryset = CustomUser.objects.all()
    serializer_class = UserRegistrationSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        return Response({'message': 'User registered successfully'}, status=status.HTTP_201_CREATED)


class UserLoginView(TokenObtainPairView):
    pass