@echo off
REM Case Study Generation Script for Windows
REM =========================================

setlocal EnableDelayedExpansion

REM Set paths (modify these if needed)
set DATA_DIR=D:\Users\11919\Documents\毕业论文\paper\code\data
set OUTPUT_DIR=D:\Users\11919\Documents\毕业论文\paper\figures

echo ========================================
echo FSD Case Study Generation
echo ========================================
echo.

REM Step 0: Validate setup
echo Step 0: Validating setup...
python test_case_study_setup.py

if errorlevel 1 (
    echo Setup validation failed. Please fix issues before proceeding.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup validated successfully!
echo ========================================
echo.

REM Step 1: Generate case studies with training
echo Step 1: Training models and generating case studies...
echo This will take 10-20 minutes with GPU, longer with CPU.
echo.

python generate_case_study.py ^
    --data_dir "%DATA_DIR%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --num_cases 3 ^
    --device cuda

if errorlevel 1 (
    echo Case study generation failed.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Case Study Generation Complete!
echo ========================================
echo.
echo Generated files in %OUTPUT_DIR%:
echo   - case_study_elliptic.pdf
echo   - case_study_attention_comparison.pdf
echo   - case_study_node_*_neighborhood.pdf
echo.
echo Models saved in %OUTPUT_DIR%:
echo   - naa_gcn_elliptic.pt
echo   - gat_elliptic.pt
echo.

REM Optional: Generate additional cases
set /p REPLY="Generate additional case studies with 5 nodes? (y/n) "
if /i "%REPLY%"=="y" (
    echo Generating 5-node case study...
    python generate_case_study.py ^
        --data_dir "%DATA_DIR%" ^
        --output_dir "%OUTPUT_DIR%\extended" ^
        --num_cases 5 ^
        --skip_training ^
        --device cuda

    echo Extended case studies saved to %OUTPUT_DIR%\extended
)

echo.
echo All done! Check %OUTPUT_DIR% for visualizations.
pause
