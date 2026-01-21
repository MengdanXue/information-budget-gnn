@echo off
echo ============================================
echo Running Missing IEEE-CIS Experiments
echo ============================================

cd /d "D:\Users\11919\Documents\毕业论文\paper\code"

echo.
echo [1/4] Running GAT (10 seeds)...
python train_ieee_cis.py --method GAT
if errorlevel 1 (
    echo GAT failed!
    pause
)

echo.
echo [2/4] Running NAA-GCN (10 seeds)...
python train_ieee_cis.py --method NAA-GCN
if errorlevel 1 (
    echo NAA-GCN failed!
    pause
)

echo.
echo [3/4] Running NAA-GAT (10 seeds)...
python train_ieee_cis.py --method NAA-GAT
if errorlevel 1 (
    echo NAA-GAT failed!
    pause
)

echo.
echo [4/4] Running MixHop (10 seeds)...
python train_ieee_cis.py --method MixHop
if errorlevel 1 (
    echo MixHop failed!
    pause
)

echo.
echo ============================================
echo All experiments completed!
echo Results saved to: results\ieee_cis_*.json
echo ============================================
pause
