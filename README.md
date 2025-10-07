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


## Source code ---> IR ##

e.g.:
(gcc -fdump-tree-gimple test.c) **Direct GCC to print GIMPLE** **—Optional—**

gcc -fprofile-arcs -ftest-coverage -o test test.c

./test

gcov ./test.c

python3 HEgcov_to_profile.py test.c.gcov profile.json

### HEgcov_to_profile.py ###

```
import sys
import re
import json
import math

ENTROPY_THRESHOLD = 0.8  # 阈值

def entropy(p):
    """Compute binary entropy in bits"""
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

def parse_gcov(gcov_file):
    branches = {}
    current_line = None
    branch_probs = []

    with open(gcov_file, "r") as f:
        for line in f:
            # 匹配源码行号，例如： "      50:    8:            if ( i % 3 == 0 )"
            m = re.match(r"\s*\d+:\s+(\d+):", line)
            if m:
                current_line = m.group(1)
                branch_probs = []  # reset for this line
                continue

            # 匹配分支覆盖信息，例如： "branch  0 taken 40%"
            m = re.search(r"branch\s+\d+\s+taken\s+(\d+)%", line)
            if m and current_line:
                branch_probs.append(int(m.group(1)))

                # 如果收集到两个分支概率，存储
                if len(branch_probs) == 2:
                    p_true = branch_probs[0] / 100.0
                    p_false = branch_probs[1] / 100.0
                    H = entropy(p_true)

                    branches[f"{gcov_file}:{current_line}:branch"] = {
                        "true_prob": p_true,
                        "false_prob": p_false,
                        "entropy": H,
                        "high_entropy": H > ENTROPY_THRESHOLD
                    }
    return branches

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 gcov_to_profile.py test.c.gcov profile.json")
        sys.exit(1)

    gcov_file = sys.argv[1]
    out_file = sys.argv[2]

    branches = parse_gcov(gcov_file)

    with open(out_file, "w") as f:
        json.dump(branches, f, indent=2)

    print(f"Wrote {len(branches)} branches to {out_file}")

if __name__ == "__main__":
    main()


```
