# PolyBench

## Functional similarity ##

To perform approximate computations on cold paths (e.g., via skip sampling or reduced step sizes) while executing full computations 
on hot paths, thereby achieving runtime acceleration in practical deployments.

## Implementation Approach ##
Apply Shannon expansion to the inner accumulation loop of the target computation with respect to a low-overhead sub-predicate **_condA_**. The branch semantics are defined as follows: if **_condA_** evaluates to true, proceed to the "full accumulation" branch under stricter conditional checks (or the more frequent/hot sub-guard); otherwise, divert to the approximate (or lighter-weight) accumulation branch. Possible approximation strategies include accumulating only a subset of the iterations (e.g., with a step size of 2) followed by proportional scaling of the result, or employing low-precision arithmetic, among others.

### Compile and run Baseline/LU/SE+LU (remember to use -DPOLYBENCH_TIME). ###
e.g. Baseline: gcc -O2 cholesky.c -I ../../../utilities ../../../utilities/polybench.c -DPOLYBENCH_TIME -lm

LU: gcc -O2 -DUNROLL=4 cholesky_o.c -I ../../../utilities ../../../utilities/polybench.c -DPOLYBENCH_TIME -lm

SE+LU:  gcc -O2 -DSE -DUNROLL=4 cholesky_o.c -I ../../../utilities ../../../utilities/polybench.c -DPOLYBENCH_TIME -lm
