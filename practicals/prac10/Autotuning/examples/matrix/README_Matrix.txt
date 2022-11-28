Autotuning System

Matrix Multiplication Example


The tutorial (in docs/tutorial.pdf) will guide you through tuning this example.

The code the tutorial begins with is in the original/ directory and the final 
code used for tuning in the tutorial is found in modified/.


There is another version, in comparison/, which performs both a naive and 
blocked multiplication and compares them. This is a common pattern in GPU code, 
where a 'gold' CPU implementation is used to check the GPU implementation being 
developed and tuned.

