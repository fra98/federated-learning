#!/usr/bin/env python3

import os

INTERPRETER = '.venv/bin/python'

ALPHAS = [0.05, 0.1, 0.2, 0.5, 1.0, 10.0, 100.0]

if __name__ == '__main__':
    for alpha in ALPHAS:
        os.system(f"{INTERPRETER} utility/split_dirichlet.py {alpha:.2f} 0 > .log/splits/split_{alpha:.2f}_dirichlet.txt")
        os.system(f"{INTERPRETER} utility/split_dirichlet.py {alpha:.2f} 1 > .log/splits/split_{alpha:.2f}_dirichlet_CB.txt")
        os.system(f"{INTERPRETER} utility/split_dirichlet.py {alpha:.2f} 2 > .log/splits/split_{alpha:.2f}_dirichlet_CB_UNBALANCED.txt")
        os.system(f"{INTERPRETER} utility/split_google.py {alpha:.2f} > .log/splits/split_{alpha:.2f}_google.txt")
