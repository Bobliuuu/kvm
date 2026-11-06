# Phased serve/train load (24 s, paced 35 Hz Doom, serve burst 8-12 s, train pause 14-17 s, scene change 18-22 s), serial vs split-tick Doom

A100-SXM4-40GB, `adaptive/adaptive_run` on this tree (executor with slot recycling + sched), 3 repetitions (seeds 1-3), 840 Doom frames per run. `exact` = every Doom frame of the run identical to the CPU oracle. Late = frame completed after its 28.57 ms deadline (`doom.deadline_misses`).

| config | reps | exact | Doom frames late (sum of 3 x 840) | frame time p50 ms | p99 ms | p999 ms | serve p50 / p99 ms | serve dropped | train steps/s |
|---|---|---|---|---|---|---|---|---|---|
| adaptive_split0 | 3 | True | 0 | 8.6 | 17.9 | 18.5 | 15.4 / 50.9 | 0 | 46.8 |
| adaptive_split8 | 3 | True | 0 | 7.4 | 16.9 | 18.9 | 15.3 / 51.3 | 0 | 46.9 |
| fixed_prio_split0 | 3 | True | 0 | 8.5 | 17.9 | 19.8 | 11.8 / 29.5 | 0 | 47.0 |
| fixed_prio_split8 | 3 | True | 0 | 7.2 | 16.9 | 18.9 | 11.8 / 29.6 | 0 | 46.9 |
| global_split0 | 3 | True | 1 | 9.0 | 19.0 | 21.1 | 150.6 / 156.1 | 1248 | 52.9 |
| global_split8 | 3 | True | 0 | 8.8 | 18.9 | 20.6 | 150.9 / 155.8 | 1251 | 52.9 |
