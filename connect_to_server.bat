@echo off
chcp 65001 >nul
echo ============================================================
echo 連接到訓練服務器
echo ============================================================
echo.

set SERVER_IP=140.118.127.87

echo 服務器 IP: %SERVER_IP%
echo.

set /p USERNAME="請輸入用戶名: "

echo.
echo 正在連接到 %USERNAME%@%SERVER_IP%...
echo.

ssh %USERNAME%@%SERVER_IP%

pause

