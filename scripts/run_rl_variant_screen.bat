@echo off
rem RL design-choice screen (Windows). Same budget per arm, one changed choice each.
setlocal enabledelayedexpansion
cd /d "%~dp0.."
if "%~1"=="" (set "OUT=runs\rl-screen") else set "OUT=%~1"
if "%~2"=="" (set "CKPT=runs\pretrain-001\checkpoint-0100") else set "CKPT=%~2"
set FAMILIES=increment,double,square,even,reverse

echo == arm baseline
run_cpu.bat scripts\train_rl.py --initial-checkpoint "%CKPT%" --output "%OUT%\baseline" --groups 16 --checkpoints 16 --group-size 8 --temperature 0.35 --reward-mode binary --families %FAMILIES% --tasks-per-family 4 --seed 31415
run_cpu.bat scripts\evaluate.py --checkpoint "%OUT%\baseline\checkpoint-0016" --split dev --per-family 4 --families %FAMILIES% --output "%OUT%\baseline\dev-eval.json"

echo == arm partial-reward
run_cpu.bat scripts\train_rl.py --initial-checkpoint "%CKPT%" --output "%OUT%\partial-reward" --groups 16 --checkpoints 16 --group-size 8 --temperature 0.35 --reward-mode case-fraction --families %FAMILIES% --tasks-per-family 4 --seed 31415
run_cpu.bat scripts\evaluate.py --checkpoint "%OUT%\partial-reward\checkpoint-0016" --split dev --per-family 4 --families %FAMILIES% --output "%OUT%\partial-reward\dev-eval.json"

echo == arm temp-020
run_cpu.bat scripts\train_rl.py --initial-checkpoint "%CKPT%" --output "%OUT%\temp-020" --groups 32 --checkpoints 32 --group-size 4 --temperature 0.20 --reward-mode binary --families %FAMILIES% --tasks-per-family 4 --seed 31415
run_cpu.bat scripts\evaluate.py --checkpoint "%OUT%\temp-020\checkpoint-0032" --split dev --per-family 4 --families %FAMILIES% --output "%OUT%\temp-020\dev-eval.json"

echo == arm temp-080
run_cpu.bat scripts\train_rl.py --initial-checkpoint "%CKPT%" --output "%OUT%\temp-080" --groups 16 --checkpoints 16 --group-size 8 --temperature 0.80 --reward-mode binary --families %FAMILIES% --tasks-per-family 4 --seed 31415
run_cpu.bat scripts\evaluate.py --checkpoint "%OUT%\temp-080\checkpoint-0016" --split dev --per-family 4 --families %FAMILIES% --output "%OUT%\temp-080\dev-eval.json"

echo == arm group-16
run_cpu.bat scripts\train_rl.py --initial-checkpoint "%CKPT%" --output "%OUT%\group-16" --groups 8 --checkpoints 8 --group-size 16 --temperature 0.35 --reward-mode binary --families %FAMILIES% --tasks-per-family 4 --seed 31415
run_cpu.bat scripts\evaluate.py --checkpoint "%OUT%\group-16\checkpoint-0008" --split dev --per-family 4 --families %FAMILIES% --output "%OUT%\group-16\dev-eval.json"

run_cpu.bat scripts\analyze_rl_variants.py --screen "%OUT%" --output "%OUT%\analysis.json"
endlocal
