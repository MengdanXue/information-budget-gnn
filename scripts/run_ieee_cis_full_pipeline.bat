@echo off
REM IEEE-CIS Full Pipeline - Windows Batch Script
REM This script runs the complete prior prediction experiment on Windows

setlocal EnableDelayedExpansion

REM Configuration
set DATA_DIR=.\ieee_cis_data
set PROCESSED_DIR=.\processed
set OUTPUT_DIR=.\prior_prediction_results
set DEVICE=cuda

REM Seeds for reproducibility
set SEEDS=42 123 456 789 1024 2048 3072 4096 5120 6144

REM Methods to evaluate
set METHODS=GCN GAT GraphSAGE H2GCN FAGCN GPRGNN NAA-GCN DAAA

echo ========================================================================
echo IEEE-CIS PRIOR PREDICTION EXPERIMENT - FULL PIPELINE
echo ========================================================================
echo.
echo This script will:
echo   1. Build graph from IEEE-CIS data
echo   2. Make FSD prediction (Phase 1)
echo   3. Run 10-seed experiments (Phase 2) - ~8-16 hours
echo   4. Generate validation report (Phase 4)
echo.
echo Configuration:
echo   Data directory:   %DATA_DIR%
echo   Output directory: %OUTPUT_DIR%
echo   Device:           %DEVICE%
echo.
set /p CONTINUE=Continue? (y/n):
if /i not "%CONTINUE%"=="y" (
    echo Aborted.
    exit /b 1
)

REM ============================================================================
REM STEP 1: Build Graph from IEEE-CIS Data
REM ============================================================================
echo.
echo ========================================================================
echo STEP 1: Building Graph from IEEE-CIS Data
echo ========================================================================
echo.

if not exist "%PROCESSED_DIR%\ieee_cis_graph.pkl" (
    echo Graph not found. Building from CSV files...

    if not exist "%DATA_DIR%" (
        echo ERROR: Data directory not found: %DATA_DIR%
        echo.
        echo Please download IEEE-CIS data from Kaggle:
        echo   https://www.kaggle.com/c/ieee-fraud-detection/data
        echo.
        echo Expected files:
        echo   %DATA_DIR%\train_transaction.csv
        echo   %DATA_DIR%\train_identity.csv
        exit /b 1
    )

    python ieee_cis_graph_builder.py ^
        --data_dir "%DATA_DIR%" ^
        --output_dir "%PROCESSED_DIR%" ^
        --max_edges_per_entity 100

    echo.
    echo [OK] Graph built successfully
) else (
    echo [OK] Graph already exists: %PROCESSED_DIR%\ieee_cis_graph.pkl
)

REM ============================================================================
REM STEP 2: Phase 1 - Prior Prediction
REM ============================================================================
echo.
echo ========================================================================
echo STEP 2: Phase 1 - Making Prior Prediction
echo ========================================================================
echo.
echo WARNING: This prediction will be timestamped and locked.
echo          After this step, you cannot modify the prediction.
echo.
set /p CONTINUE2=Continue? (y/n):
if /i not "%CONTINUE2%"=="y" (
    echo Aborted.
    exit /b 1
)

python prior_prediction_experiment.py ^
    --phase 1 ^
    --data_path "%PROCESSED_DIR%\ieee_cis_graph.pkl" ^
    --output_dir "%OUTPUT_DIR%"

echo.
echo [OK] Prediction committed with timestamp
echo.
pause

REM ============================================================================
REM STEP 3: Phase 2 - Experimental Validation
REM ============================================================================
echo.
echo ========================================================================
echo STEP 3: Phase 2 - Running Experiments (This will take 8-16 hours)
echo ========================================================================
echo.
echo Running: %METHODS%
echo With seeds: %SEEDS%
echo On device: %DEVICE%
echo.
echo Estimated time: 8-16 hours (GPU) or 24-48 hours (CPU)
echo.
set /p CONTINUE3=Start experiments? (y/n):
if /i not "%CONTINUE3%"=="y" (
    echo Aborted. You can run experiments later with:
    echo   python prior_prediction_experiment.py --phase 2
    exit /b 0
)

REM Start time
for /f "tokens=1-4 delims=:.," %%a in ("%time%") do (
    set /a "start=(((%%a*60)+1%%b %% 100)*60+1%%c %% 100)*100+1%%d %% 100"
)

python prior_prediction_experiment.py ^
    --phase 2 ^
    --data_path "%PROCESSED_DIR%\ieee_cis_graph.pkl" ^
    --output_dir "%OUTPUT_DIR%" ^
    --methods %METHODS% ^
    --seeds %SEEDS% ^
    --device "%DEVICE%"

REM End time
for /f "tokens=1-4 delims=:.," %%a in ("%time%") do (
    set /a "end=(((%%a*60)+1%%b %% 100)*60+1%%c %% 100)*100+1%%d %% 100"
)
set /a elapsed=end-start
set /a hours=elapsed / 360000
set /a minutes=(elapsed %% 360000) / 6000

echo.
echo [OK] All experiments completed
echo    Time elapsed: %hours%h %minutes%m

REM ============================================================================
REM STEP 4: Phase 4 - Validation Report
REM ============================================================================
echo.
echo ========================================================================
echo STEP 4: Phase 4 - Generating Validation Report
echo ========================================================================
echo.

python prior_prediction_experiment.py ^
    --phase 4 ^
    --output_dir "%OUTPUT_DIR%"

echo.
echo [OK] Validation report generated

REM ============================================================================
REM FINAL SUMMARY
REM ============================================================================
echo.
echo ========================================================================
echo EXPERIMENT COMPLETE
echo ========================================================================
echo.
echo Output files:
echo   Prediction:  %OUTPUT_DIR%\fsd_prediction.json
echo   Results:     %OUTPUT_DIR%\experimental_results.json
echo   Report:      %OUTPUT_DIR%\validation_report.md
echo.
echo Next steps:
echo   1. Review the validation report
echo   2. Check if FSD prediction was correct
echo   3. Include results in your paper
echo.
echo ========================================================================

pause
endlocal
