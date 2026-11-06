# Runs (Modal A100, detached apps; GPU as printed in each job's gpu.txt / start event; no Lambda lock used)
| time (UTC) | app | job | what | result dir |
|---|---|---|---|---|
| 2026-11-06 07:06 | ap-l1uz5PjdjLeMTijBMuGog9 | diag | baseline build, mlp_ce/xfmr_odd timing, --blocks sweep, trace + hop analysis (A100-SXM4-80GB 1410 MHz) | modal-diag/diag |
| 2026-11-06 08:00 | ap-4XRidq4eM5lJIyFOShKnLc | win | record size sweep (live_cap/ring_words), baseline binary (A100-SXM4-80GB 1410 MHz) | modal-win/win |
| 2026-11-06 08:50 | ap-phvrGx4BtciTNTHz7qgHPZ | trace1w | one-record trace + analysis (cached baseline binary) | modal-trace1w/trace1w |
| 2026-11-06 08:55 | ap-N6C09httcyukjSZofFUKvM | ab | first run-next A/B + test_exec (A100-SXM4-40GB) | (pending) |
| 2026-11-06 08:57 | ap-5ulSAvsFPNVn3N5QaSOxAo | round2 | autosize base vs run-next / poll variants, md5, test_exec | (pending) |
