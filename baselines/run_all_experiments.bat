@echo off
REM Batch script to run all baseline comparison experiments
REM Author: FSD-GNN Paper
REM Date: 2024-12-23

echo ================================================================================
echo 2024 SOTA Baseline Comparison Experiments
echo ================================================================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8+
    exit /b 1
)

echo Starting experiments...
echo.

REM Run all baselines on IEEE-CIS
echo [1/4] Running baselines on IEEE-CIS dataset...
python run_baselines.py --model ARC GAGA CARE-GNN PC-GNN VecAug SEFraud --dataset ieee-cis --seeds 42 123 456 789 1024

echo.
REM Run all baselines on YelpChi
echo [2/4] Running baselines on YelpChi dataset...
python run_baselines.py --model ARC GAGA CARE-GNN PC-GNN VecAug SEFraud --dataset yelpchi --seeds 42 123 456 789 1024

echo.
REM Run FSD-GNN models on IEEE-CIS
echo [3/4] Running FSD-GNN models on IEEE-CIS dataset...
python run_baselines.py --model DAAA DAAAv2 DAAAv3 DAAAv4 --dataset ieee-cis --seeds 42 123 456 789 1024

echo.
REM Run FSD-GNN models on YelpChi
echo [4/4] Running FSD-GNN models on YelpChi dataset...
python run_baselines.py --model DAAA DAAAv2 DAAAv3 DAAAv4 --dataset yelpchi --seeds 42 123 456 789 1024

echo.
echo ================================================================================
echo All experiments completed!
echo Results saved in ./results/ directory
echo ================================================================================
pause
