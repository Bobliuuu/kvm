# The base against this branch on the read's composition: one executor (6bf2bc1b: main with #26, #28, #32, #35, #40, #44, the gang stack
# through #46-#48 and the K park), its frontend without #39. A100-SXM4-80GB, Nov 11. Every weight buffer gated bitwise.
# Read 14:40-14:57Z on this branch before AdamW's chunks were sized to the model (flat 16,384-element chunks), then 15:25-15:28Z on the
# three bundles the sizing changes ("sized"); the other 28 bundles are byte-identical either way.

## The panel, 5 paired rounds
| row | base | this branch | speed | rounds | tasks | weights |
|---|---|---|---|---|---|---|
| gpt-tiny | 6.62 ms | 6.62 ms | x1.000 (x0.999-1.002) | 5 | 6,749 → 6,799 | bitwise |
| gpt-small | 14.33 ms | 14.33 ms | x1.000 (x0.999-1.002) | 5 | 28,858 → 29,158 | bitwise |
| gpt-162m | 302.90 ms | 306.44 ms | x0.989 (x0.986-0.992) | 5 | 161,316 → 191,045 | bitwise |
| mlp1024-r1 | 6.79 ms | 6.79 ms | x0.999 (x0.997-1.003) | 5 | 17,117 → 18,317 | bitwise |
| mlp256-r1 | 4.73 ms | 4.73 ms | x0.999 (x0.994-1.002) | 5 | 8,394 → 8,994 | bitwise |
| mlp72-r1 | 5.60 ms | 5.61 ms | x0.999 (x0.997-1.003) | 5 | 9,600 | bitwise |
| mlp-hetero-r8 | 6.28 ms | 6.28 ms | x1.000 (x0.999-1.001) | 5 | 68,400 → 68,800 | bitwise |
| mlp72-r32 | 8.26 ms | 8.26 ms | x1.000 (x0.999-1.001) | 5 | 307,200 | bitwise |
| mlp72-r8 | 5.51 ms | 5.51 ms | x0.999 (x0.999-0.999) | 5 | 76,800 | bitwise |

## gpt-162m and two hold-out shapes with the sized chunks, 5 paired rounds
| row | base | this branch (sized) | speed | rounds | tasks | weights |
|---|---|---|---|---|---|---|
| w_162m_3 | 303.82 ms | 297.10 ms | x1.023 (x1.014-1.024) | 5 | 161,316 → 165,017 | bitwise |
| w_ho-gpt-d384-h6-l4-t128-b4-v512 | 141.05 ms | 120.28 ms | x1.172 (x1.171-1.173) | 5 | 83,750 → 101,622 | bitwise |
| w_ho-gpt-d512-h8-l4-t32-b1-v128 | 74.54 ms | 61.89 ms | x1.205 (x1.203-1.205) | 5 | 58,824 → 78,977 | bitwise |

## The two read draws, 5 paired rounds
| row | base | this branch | speed | rounds | tasks | weights |
|---|---|---|---|---|---|---|
| ho-d384-l3-t96-b4 | 101.64 ms | 86.62 ms | x1.173 (x1.172-1.175) | 5 | 57,618 → 71,625 | bitwise |
| ho-d512-l1-t64-b2 | 42.02 ms | 27.73 ms | x1.514 (x1.513-1.530) | 5 | 22,748 → 30,843 | bitwise |

## The 12 hold-out shapes at 30 steps, 1 round each (flat chunks)
| row | base | this branch | speed | rounds | tasks | weights |
|---|---|---|---|---|---|---|
| ho-gpt-d192-h3-l1-t32-b1-v64 | 14.48 ms | 9.61 ms | x1.507 (x1.507-1.507) | 1 | 7,112 → 8,703 | bitwise |
| ho-gpt-d192-h3-l2-t64-b4-v256 | 32.79 ms | 28.41 ms | x1.154 (x1.154-1.154) | 1 | 29,913 → 33,195 | bitwise |
| ho-gpt-d192-h3-l4-t128-b2-v512 | 69.79 ms | 53.91 ms | x1.294 (x1.294-1.294) | 1 | 54,457 → 61,021 | bitwise |
| ho-gpt-d256-h4-l1-t128-b4-v512 | 46.08 ms | 36.42 ms | x1.265 (x1.265-1.265) | 1 | 19,108 → 21,877 | bitwise |
| ho-gpt-d256-h4-l2-t32-b2-v128 | 34.55 ms | 22.93 ms | x1.507 (x1.507-1.507) | 1 | 20,222 → 24,884 | bitwise |
| ho-gpt-d256-h4-l3-t96-b4-v256 | 60.51 ms | 48.83 ms | x1.239 (x1.239-1.239) | 1 | 64,565 → 71,606 | bitwise |
| ho-gpt-d384-h6-l1-t64-b1-v64 | 27.27 ms | 17.60 ms | x1.550 (x1.550-1.550) | 1 | 12,781 → 17,530 | bitwise |
| ho-gpt-d384-h6-l3-t96-b4-v256 | 101.78 ms | 86.66 ms | x1.174 (x1.174-1.174) | 1 | 57,618 → 71,625 | bitwise |
| ho-gpt-d384-h6-l4-t128-b4-v512 | 140.87 ms | 120.28 ms | x1.171 (x1.171-1.171) | 1 | 83,750 → 102,872 | bitwise |
| ho-gpt-d512-h8-l1-t64-b2-v256 | 42.22 ms | 27.73 ms | x1.522 (x1.522-1.522) | 1 | 22,748 → 30,843 | bitwise |
| ho-gpt-d512-h8-l2-t128-b4-v512 | 115.95 ms | 100.03 ms | x1.159 (x1.159-1.159) | 1 | 47,483 → 63,321 | bitwise |
| ho-gpt-d512-h8-l4-t32-b1-v128 | 74.48 ms | 62.03 ms | x1.201 (x1.201-1.201) | 1 | 58,824 → 88,997 | bitwise |

## The executor's path for each bundle (one plain run on the same binary; its rule: static at a mean estimated task of 6.0 us or less)
base ho/w_ho-gpt-d384-h6-l3-t96-b4 | path: dynamic (mean estimated task 6.09 us > 6.0
this branch, flat ho/w_ho-gpt-d384-h6-l3-t96-b4 | path: static (mean estimated task 5.74 us <= 6.0
base ho/w_ho-gpt-d512-h8-l1-t64-b2 | path: static (mean estimated task 5.93 us <= 6.0
this branch, flat ho/w_ho-gpt-d512-h8-l1-t64-b2 | path: static (mean estimated task 5.51 us <= 6.0
base panel/w_162m_3 | path: dynamic (mean estimated task 10.67 us > 6.0
this branch, flat panel/w_162m_3 | path: dynamic (mean estimated task 9.67 us > 6.0
this branch, sized panel/w_162m_3 | path: dynamic (mean estimated task 10.48 us > 6.0
base tsho/w_ho-gpt-d384-h6-l4-t128-b4-v512 | path: dynamic (mean estimated task 6.27 us > 6.0
this branch, flat tsho/w_ho-gpt-d384-h6-l4-t128-b4-v512 | path: static (mean estimated task 5.90 us <= 6.0
this branch, sized tsho/w_ho-gpt-d384-h6-l4-t128-b4-v512 | path: static (mean estimated task 5.91 us <= 6.0
base tsho/w_ho-gpt-d512-h8-l4-t32-b1-v128 | path: static (mean estimated task 4.82 us <= 6.0
this branch, flat tsho/w_ho-gpt-d512-h8-l4-t32-b1-v128 | path: static (mean estimated task 4.64 us <= 6.0
this branch, sized tsho/w_ho-gpt-d512-h8-l4-t32-b1-v128 | path: static (mean estimated task 4.60 us <= 6.0
