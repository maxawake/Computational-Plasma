#/bin/bash
mpicxx main.cpp -o main -lm && mpirun -np 4 ./main && python eval.py
