# main's binary (ee481fa1) against this branch's (f82f1923), each on its own default rule, 5 paired rounds a bundle, every weight buffer bitwise.
| row | main | this branch | speed | rounds | tasks | weights |
|---|---|---|---|---|---|---|
| ho-gpt-d192-h3-l1-t32-b1-v64 | 18.85 ms | 18.86 ms | x1.000 (x0.998-1.002) | 5 | 7,112 | bitwise |
| ho-gpt-d192-h3-l2-t64-b4-v256 | 41.87 ms | 41.87 ms | x1.000 (x0.998-1.000) | 5 | 29,913 | bitwise |
| ho-gpt-d192-h3-l4-t128-b2-v512 | 74.56 ms | 74.56 ms | x1.000 (x0.999-1.001) | 5 | 54,457 | bitwise |
| ho-gpt-d256-h4-l1-t128-b4-v512 | 51.73 ms | 51.70 ms | x1.001 (x0.999-1.002) | 5 | 19,108 | bitwise |
| ho-gpt-d256-h4-l2-t32-b2-v128 | 45.99 ms | 45.99 ms | x1.000 (x0.998-1.001) | 5 | 20,222 | bitwise |
| ho-gpt-d256-h4-l3-t96-b4-v256 | 75.29 ms | 75.27 ms | x1.000 (x0.999-1.001) | 5 | 64,565 | bitwise |
| ho-gpt-d384-h6-l1-t64-b1-v64 | 34.87 ms | 34.88 ms | x1.001 (x0.999-1.001) | 5 | 12,781 | bitwise |
| ho-gpt-d384-h6-l3-t96-b4-v256 | 108.94 ms | 102.97 ms | x1.058 (x1.057-1.060) | 5 | 57,618 | bitwise |
| ho-gpt-d384-h6-l4-t128-b4-v512 | 151.13 ms | 142.80 ms | x1.058 (x1.057-1.059) | 5 | 83,750 | bitwise |
| ho-gpt-d512-h8-l1-t64-b2-v256 | 51.88 ms | 51.87 ms | x1.000 (x1.000-1.001) | 5 | 22,748 | bitwise |
| ho-gpt-d512-h8-l2-t128-b4-v512 | 121.21 ms | 119.57 ms | x1.013 (x1.013-1.015) | 5 | 47,483 | bitwise |
| ho-gpt-d512-h8-l4-t32-b1-v128 | 94.79 ms | 94.71 ms | x1.001 (x0.999-1.001) | 5 | 58,824 | bitwise |
| gpt-d640-h10-l2-t128-b4-v512 | 44.41 ms | 44.43 ms | x1.000 (x0.995-1.002) | 5 | 19,863 | bitwise |
| gpt-d512-h8-l2-t128-b8-v512 | 48.36 ms | 48.57 ms | x0.997 (x0.994-1.004) | 5 | 24,113 | bitwise |
| gpt-d768-h12-l4-t128-b4-v512 | 85.41 ms | 85.08 ms | x1.005 (x0.988-1.027) | 5 | 45,833 | bitwise |
| gpt-d1024-h16-l2-t128-b4-v512 | 64.33 ms | 64.25 ms | x1.002 (x0.993-1.005) | 5 | 30,687 | bitwise |
| gpt-d640-h10-l2-t256-b4-v512 | 60.70 ms | 60.75 ms | x1.000 (x0.995-1.007) | 5 | 28,675 | bitwise |
| gpt-d768-h12-l2-t256-b4-v512 | 73.10 ms | 73.04 ms | x1.003 (x0.999-1.014) | 5 | 34,509 | bitwise |
| gpt-d1024-h16-l2-t256-b4-v512 | 127.86 ms | 128.58 ms | x1.002 (x0.991-1.036) | 5 | 37,641 | bitwise |
| gpt-d1024-h16-l2-t256-b8-v512 | 168.02 ms | 167.96 ms | x1.000 (x0.998-1.002) | 5 | 60,501 | bitwise |
| gpt-d768-h12-l2-t512-b8-v512 | 214.36 ms | 214.44 ms | x0.998 (x0.998-1.003) | 5 | 79,909 | bitwise |
| gpt-d1536-h24-l2-t256-b4-v512 | 189.65 ms | 190.34 ms | x0.999 (x0.994-1.000) | 5 | 61,383 | bitwise |
| gpt-162m | 348.95 ms | 348.43 ms | x1.003 (x0.996-1.005) | 5 | 161,316 | bitwise |
| h640-d4-r16-generic | 78.97 ms | 78.95 ms | x1.000 (x0.999-1.001) | 5 | 307,200 | bitwise |
