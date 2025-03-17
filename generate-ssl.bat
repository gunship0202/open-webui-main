@echo off
chcp 437
mkdir ssl 2>nul

REM Generate private key and certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout ssl/private.key -out ssl/certificate.crt -subj "/CN=10.0.239.87/O=Local Development/C=TW"

echo Certificate generation completed!
echo Certificate location: ssl/certificate.crt
echo Private key location: ssl/private.key

pause 