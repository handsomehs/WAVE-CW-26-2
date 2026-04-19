# Dual-Queue Boundary Split Repeats

- Updated: 2026-04-18T18:01:12Z
- Namespace: eidf018ns
- Repeats: 3
- Pairs: a100-4g,h100-8g

| pair | repeat | pq_job | uq_job | winner | winner_dir | action |
|---|---:|---|---|---|---|---|
| a100-4g | 1 | - | - | uq | logs/awave-cw2-boundary-ab-a100-4g-uq-r1 | skip_existing_valid |
| a100-4g | 2 | awave-cw2-boundary-ab-a100-4g-pq-r2-nk6xl | awave-cw2-boundary-ab-a100-4g-uq-r2-nz9wb | uq | logs/awave-cw2-boundary-ab-a100-4g-uq-r2 | deleted_loser_job=awave-cw2-boundary-ab-a100-4g-pq-r2-nk6xl |
| a100-4g | 3 | awave-cw2-boundary-ab-a100-4g-pq-r3-zk86d | awave-cw2-boundary-ab-a100-4g-uq-r3-gjbzh | uq | logs/awave-cw2-boundary-ab-a100-4g-uq-r3 | deleted_loser_job=awave-cw2-boundary-ab-a100-4g-pq-r3-zk86d |
| h100-8g | 1 | - | - | pq | logs/awave-cw2-boundary-ab-h100-8g-pq-r1 | skip_existing_valid |
| h100-8g | 2 | awave-cw2-boundary-ab-h100-8g-pq-r2-tght8 | awave-cw2-boundary-ab-h100-8g-uq-r2-2dt89 | pq | logs/awave-cw2-boundary-ab-h100-8g-pq-r2 | deleted_loser_job=awave-cw2-boundary-ab-h100-8g-uq-r2-2dt89 |
| h100-8g | 3 | awave-cw2-boundary-ab-h100-8g-pq-r3-krdxx | awave-cw2-boundary-ab-h100-8g-uq-r3-czffl | pq | logs/awave-cw2-boundary-ab-h100-8g-pq-r3 | deleted_loser_job=awave-cw2-boundary-ab-h100-8g-uq-r3-czffl |
