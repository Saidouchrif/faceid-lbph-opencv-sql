@echo off
REM Script pour installer les dépendances Python localement

echo [INFO] Installation des dépendances Python...
python -m pip install --upgrade pip
python -m pip install opencv-contrib-python==4.9.0.80
python -m pip install numpy==1.26.4
python -m pip install mysql-connector-python==8.3.0
python -m pip install pillow==10.3.0
python -m pip install imutils==0.5.4

echo [OK] Installation terminée!
echo.
echo Vous pouvez maintenant lancer: python recognize_camera_local.py
pause
