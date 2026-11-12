NVIDIA GeForce RTX 5090, Intel(R) Core(TM) Ultra 9 285K, 68.1 GB RAM; PyTomography 4.0.0, PyTorch 2.14.1+cu130

| Case | Step | First run (s) | Repeats (s, median) | Peak RAM (GB) | Peak GPU (GB) |
|---|---|---|---|---|---|
| pct_abdomen/fbp | load | 2.31 |  | 12.4 | 0.00 |
| pct_abdomen/fbp | FBP, two passes | 11.67 | 10.89 | 15.4 | 1.88 |
| pct_abdomen/sart | load | 2.24 |  | 12.4 | 0.00 |
| pct_abdomen/sart | FBP, two passes (the starting image) | 11.65 |  | 15.4 | 1.88 |
| pct_abdomen/sart | SART, 4 passes | 103.39 | 102.539 | 16.4 | 2.74 |
