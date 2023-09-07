from django.urls import include, path
from .views import UserLoginView, UserRegistrationView
# from allauth.account.views import PasswordResetConfirmView
from rest_framework_simplejwt.views import TokenBlacklistView


urlpatterns = [
    path('register/', UserRegistrationView.as_view(), name='register'),
    path('token/', UserLoginView.as_view(), name='token_obtain_pair'),
    path('token/blacklist/', TokenBlacklistView.as_view(), name='token_blacklist'),
    # path('accounts/', include('allauth.urls')),
    # path('accounts/reset/<uidb64>/<token>/', PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
]
