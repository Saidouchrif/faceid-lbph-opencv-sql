@echo off
REM Script pour lancer la reconnaissance faciale avec la caméra

echo ========================================
echo Reconnaissance Faciale - LBPH + OpenCV
echo ========================================
echo.

REM Vérifier si Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installé ou n'est pas dans le PATH
    echo Veuillez installer Python 3.11+ depuis https://www.python.org/
    pause
    exit /b 1
)

REM Vérifier si le modèle existe
if not exist "data\model.yml" (
    echo [ERREUR] Le modèle n'a pas été trouvé: data\model.yml
    echo.
    echo Veuillez d'abord exécuter:
    echo   docker-compose exec app python import_people_mysql.py
    echo.
    pause
    exit /b 1
)

REM Vérifier si les dépendances sont installées
python -c "import cv2" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installation des dépendances Python...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERREUR] Impossible d'installer les dépendances
        pause
        exit /b 1
    )
)

echo [OK] Lancement de la reconnaissance faciale...
echo.
echo Contrôles:
echo   - Q ou ESC: Quitter
echo   - C: Changer de caméra
echo.

python recognize_camera_local.py

pause
