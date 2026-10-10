@echo off
rem Full recursive self-improvement run: pretrain (skipped if present) then RSI ladder.
setlocal
cd /d "%~dp0.."
if "%~1"=="" (set "RUN=runs\self-improve-001") else set "RUN=%~1"
if "%~2"=="" (set "CHECKPOINT=runs\pretrain-001\checkpoint-0100") else set "CHECKPOINT=%~2"
if not exist "%CHECKPOINT%" (
  echo checkpoint %CHECKPOINT% missing; pretraining first
  run_cpu.bat scripts\train_pretrain.py --output runs\pretrain-001 --steps 150 ^
    --checkpoints 50,100,150 --batch-size 16 --sequence-length 128 --seed 42
)
run_cpu.bat scripts\train_self_improve.py --initial-checkpoint "%CHECKPOINT%" --output "%RUN%" ^
  --cycles 3 --proposals-per-task 16 --temperature 0.8 ^
  --fine-tune-steps 24 --fine-tune-batch-size 8 --learning-rate 1e-4 ^
  --families-ladder increment,double --per-family 4 --eval-per-family 2 --seed 73
endlocal
