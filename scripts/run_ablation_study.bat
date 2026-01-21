@echo off
REM Batch script to run NAA ablation study on all datasets
REM Usage: run_ablation_study.bat

echo ========================================
echo NAA Ablation Study - Full Pipeline
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Set default seeds (5 seeds for reasonable runtime)
set SEEDS=42 123 456 789 1024

REM Optional: Use 10 seeds for more robust results (slower)
REM set SEEDS=42 123 456 789 1024 2048 3072 4096 5120 6144

echo Using seeds: %SEEDS%
echo.

REM ========================================
REM Elliptic Dataset
REM ========================================
if exist "processed\elliptic_graph.pkl" (
    echo ========================================
    echo Running ablation study on Elliptic
    echo ========================================
    python ablation_study.py ^
        --data_path processed\elliptic_graph.pkl ^
        --dataset_name "Elliptic" ^
        --seeds %SEEDS% ^
        --output_dir ablation_results\elliptic

    if errorlevel 1 (
        echo ERROR: Ablation study failed on Elliptic
        pause
        exit /b 1
    )
    echo.
    echo Elliptic ablation completed successfully!
    echo.
) else (
    echo WARNING: Elliptic data not found at processed\elliptic_graph.pkl
    echo Skipping Elliptic ablation study.
    echo.
)

REM ========================================
REM IEEE-CIS Dataset
REM ========================================
if exist "processed\ieee_cis_graph.pkl" (
    echo ========================================
    echo Running ablation study on IEEE-CIS
    echo ========================================
    python ablation_study.py ^
        --data_path processed\ieee_cis_graph.pkl ^
        --dataset_name "IEEE-CIS" ^
        --seeds %SEEDS% ^
        --output_dir ablation_results\ieee_cis

    if errorlevel 1 (
        echo ERROR: Ablation study failed on IEEE-CIS
        pause
        exit /b 1
    )
    echo.
    echo IEEE-CIS ablation completed successfully!
    echo.
) else (
    echo WARNING: IEEE-CIS data not found at processed\ieee_cis_graph.pkl
    echo Skipping IEEE-CIS ablation study.
    echo.
)

REM ========================================
REM YelpChi Dataset
REM ========================================
if exist "processed\yelpchi_graph.pkl" (
    echo ========================================
    echo Running ablation study on YelpChi
    echo ========================================
    python ablation_study.py ^
        --data_path processed\yelpchi_graph.pkl ^
        --dataset_name "YelpChi" ^
        --seeds %SEEDS% ^
        --output_dir ablation_results\yelpchi

    if errorlevel 1 (
        echo ERROR: Ablation study failed on YelpChi
        pause
        exit /b 1
    )
    echo.
    echo YelpChi ablation completed successfully!
    echo.
) else (
    echo WARNING: YelpChi data not found at processed\yelpchi_graph.pkl
    echo Skipping YelpChi ablation study.
    echo.
)

REM ========================================
REM Amazon Dataset
REM ========================================
if exist "processed\amazon_graph.pkl" (
    echo ========================================
    echo Running ablation study on Amazon
    echo ========================================
    python ablation_study.py ^
        --data_path processed\amazon_graph.pkl ^
        --dataset_name "Amazon" ^
        --seeds %SEEDS% ^
        --output_dir ablation_results\amazon

    if errorlevel 1 (
        echo ERROR: Ablation study failed on Amazon
        pause
        exit /b 1
    )
    echo.
    echo Amazon ablation completed successfully!
    echo.
) else (
    echo WARNING: Amazon data not found at processed\amazon_graph.pkl
    echo Skipping Amazon ablation study.
    echo.
)

REM ========================================
REM Summary
REM ========================================
echo.
echo ========================================
echo ABLATION STUDY COMPLETE
echo ========================================
echo.
echo Results saved to ablation_results\
echo.
echo Generated files for each dataset:
echo   - ablation_table.tex          (Component ablation LaTeX table)
echo   - lambda_sensitivity_table.tex (Lambda sensitivity LaTeX table)
echo   - ablation_results.json        (Complete results in JSON)
echo.

REM Check if any results exist
if exist "ablation_results\" (
    echo Datasets processed:
    if exist "ablation_results\elliptic\" echo   - Elliptic
    if exist "ablation_results\ieee_cis\" echo   - IEEE-CIS
    if exist "ablation_results\yelpchi\" echo   - YelpChi
    if exist "ablation_results\amazon\" echo   - Amazon
) else (
    echo WARNING: No ablation results found!
    echo Please check that dataset files exist in processed\ directory.
)

echo.
echo Next steps:
echo 1. Review LaTeX tables in ablation_results\{dataset}\*.tex
echo 2. Copy tables to your paper LaTeX source
echo 3. Analyze results in ablation_results\{dataset}\ablation_results.json
echo.

pause
