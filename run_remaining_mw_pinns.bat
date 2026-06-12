@echo off
set PYTHON_EXE=c:\Users\acer\Desktop\wpinns\venv\Scripts\python.exe

echo "Running Helmholtz"
cd c:\Users\acer\Desktop\wpinns\MW-PINN\Helmholtz
%PYTHON_EXE% main.py > output.log 2>&1

echo "Running Maxwell_Heterogeneous"
cd c:\Users\acer\Desktop\wpinns\MW-PINN\Maxwell_Heterogeneous
%PYTHON_EXE% main.py > output.log 2>&1

echo "Running SPP_Ex1"
cd c:\Users\acer\Desktop\wpinns\MW-PINN\SPP_Ex1
%PYTHON_EXE% main.py > output.log 2>&1

echo "Running Lid_Driven_Re100"
cd c:\Users\acer\Desktop\wpinns\MW-PINN\Lid_Driven_Re100
%PYTHON_EXE% main.py > output.log 2>&1

echo "All remaining MW-PINN models completed."
