@echo off
rem Research suite: data diversity, ordering/curriculum, vision comparison.
rem Results land under artifacts\experiments and artifacts\receipts.
setlocal
cd /d "%~dp0.."
if "%~1"=="" (set "SEEDS=11,22") else set "SEEDS=%~1"
if "%~2"=="" (set "STEPS=60") else set "STEPS=%~2"

echo == data diversity
run_cpu.bat experiments\pretraining_data_diversity.py --output artifacts\experiments\pretraining-data-diversity --steps %STEPS% --batch-size 16 --seeds %SEEDS%
run_cpu.bat experiments\analyze_pretraining_data_diversity.py --results artifacts\experiments\pretraining-data-diversity\results.json --output artifacts\experiments\pretraining-data-diversity\analysis.json

echo == ordering and curriculum
set /a OSTEPS=%STEPS%*2
run_cpu.bat experiments\pretraining_curriculum_order.py --output artifacts\experiments\pretraining-ordering --steps %OSTEPS% --batch-size 16 --seeds %SEEDS%
run_cpu.bat experiments\analyze_pretraining_curriculum_order.py --results artifacts\experiments\pretraining-ordering\results.json --output artifacts\experiments\pretraining-ordering\analysis.json

echo == vision architecture comparison
run_cpu.bat scripts\train_vision.py --architecture both --device auto --seed 42 --steps 120 --batch-size 20 --eval-examples 200 --learning-rate 0.002 --output artifacts\receipts\vision-miniature-rgb.json

echo == done
endlocal
