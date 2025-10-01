Setup: pip install pennylane scikit-learn matplotlib

run: python quantum_vs_classical_shapes.py


- if pennylane shows a yellow underline, then that is probably because you need a virtual enviornment, create a virtual enviornment respective to your operating system (or use what you have, I am personally using a conda enviornment),

- to change python interpreter do ctrl p this is what I do on windows or whatever it is respective to your os

- I am running this on python 3.13.5 for anyone who is curious but anything 3.11+ should work fine

- if this seems like it is taking forever that is because the plotting step is doing 400,000 quantum evaluations so each grid point (200 by 200) calls a circut so it crawls

- this is quite a bit heavy on ram and cpu but should work fine on your laptops 

- after running this took about 10~15 minutes