# Deliverability battery (offline acceptance: timing + box + b round-trip + SAFE PNS)

budget t_excite=3 t_refocus=6 t_readout_pre_echo=14 ms; TE=60 ms; n_restarts=48.
PNS = pypulseq SAFE on the exported .seq (100%=limit, ~80%=normal mode). pns_target=65 in-design (margin for the coarse-grid underestimate).

| pulse | scanner | mode | b (s/mm2) | timing | maxG/lim | slew/lim | b_rel_err | SAFE PNS (export) | .seq |
|---|---|---|---|---|---|---|---|---|---|
| LTE | Prisma | baseline | 2252 | OK | 75/80 | 169/200 | 0.7% | 113% OVER | seq_LTE_prisma_baseline.seq |
| LTE | Prisma | pns-aware | 2756 | OK | 80/80 | 176/200 | 1.1% | 86% 1st | seq_LTE_prisma_pns-aware.seq |
| PTE | Prisma | baseline | 902 | OK | 79/80 | 160/200 | 1.1% | 109% OVER | seq_PTE_prisma_baseline.seq |
| PTE | Prisma | pns-aware | 459 | OK | 79/80 | 170/200 | 1.2% | 90% 1st | seq_PTE_prisma_pns-aware.seq |
| STE | Prisma | baseline | 444 | OK | 79/80 | 156/200 | 0.3% | 137% OVER | seq_STE_prisma_baseline.seq |
| STE | Prisma | pns-aware | 540 | OK | 79/80 | 174/200 | 0.6% | 94% 1st | seq_STE_prisma_pns-aware.seq |
| OGSE | Prisma | baseline | 54 | OK | 77/80 | 176/200 | 0.7% | 137% OVER | seq_OGSE_prisma_baseline.seq |
| OGSE | Prisma | pns-aware | 157 | OK | 80/80 | 171/200 | 0.5% | 101% OVER | seq_OGSE_prisma_pns-aware.seq |
| LTE | Connectom | baseline | 25335 | OK | 229/300 | 148/200 | 1.7% | 223% OVER | seq_LTE_connectom_baseline.seq |
| LTE | Connectom | pns-aware | 54323 | OK | 292/300 | 180/200 | 1.1% | 119% OVER | seq_LTE_connectom_pns-aware.seq |
| PTE | Connectom | baseline | 10130 | OK | 297/300 | 182/200 | 1.0% | 144% OVER | seq_PTE_connectom_baseline.seq |
| PTE | Connectom | pns-aware | 9531 | OK | 293/300 | 165/200 | 1.2% | 108% OVER | seq_PTE_connectom_pns-aware.seq |
| STE | Connectom | baseline | 4702 | OK | 288/300 | 179/200 | 1.1% | 147% OVER | seq_STE_connectom_baseline.seq |
| STE | Connectom | pns-aware | 5464 | OK | 285/300 | 195/200 | 1.0% | 117% OVER | seq_STE_connectom_pns-aware.seq |
| OGSE | Connectom | baseline | 402 | OK | 284/300 | 196/200 | 0.2% | 300% OVER | seq_OGSE_connectom_baseline.seq |
| OGSE | Connectom | pns-aware | 2844 | OK | 282/300 | 183/200 | 0.1% | 117% OVER | seq_OGSE_connectom_pns-aware.seq |
