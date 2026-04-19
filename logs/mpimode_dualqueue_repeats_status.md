# Dual-Queue MPI Mode Repeats

- Updated: 2026-04-18T15:16:47Z
- Namespace: eidf018ns
- Repeats: 5
- Pairs: a100-2g

| pair | repeat | pq_job | uq_job | winner | winner_dir | action |
|---|---:|---|---|---|---|---|
| a100-2g | 1 | - | - | pq | logs/awave-cw2-mpimode-ab-a100-2g-pq-r1 | skip_existing_valid |
| a100-2g | 2 | - | - | pq | logs/awave-cw2-mpimode-ab-a100-2g-pq-r2 | skip_existing_valid |
| a100-2g | 3 | - | - | uq | logs/awave-cw2-mpimode-ab-a100-2g-uq-r3 | skip_existing_valid |
| a100-2g | 4 | awave-cw2-mpimode-ab-a100-2g-pq-r4-gq9xl | awave-cw2-mpimode-ab-a100-2g-uq-r4-rzv86 | uq | logs/awave-cw2-mpimode-ab-a100-2g-uq-r4 | deleted_loser_job=awave-cw2-mpimode-ab-a100-2g-pq-r4-gq9xl |
| a100-2g | 5 | awave-cw2-mpimode-ab-a100-2g-pq-r5-9tftb | awave-cw2-mpimode-ab-a100-2g-uq-r5-8zkqz | uq | logs/awave-cw2-mpimode-ab-a100-2g-uq-r5 | deleted_loser_job=awave-cw2-mpimode-ab-a100-2g-pq-r5-9tftb |
