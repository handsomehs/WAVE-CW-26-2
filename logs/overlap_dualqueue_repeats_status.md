# Dual-Queue Overlap Repeats

- Updated: 2026-04-18T16:10:49Z
- Namespace: eidf018ns
- Repeats: 5
- Pairs: a100-2g,h100-2g

| pair | repeat | pq_job | uq_job | winner | winner_dir | action |
|---|---:|---|---|---|---|---|
| a100-2g | 1 | - | - | pq | logs/awave-cw2-overlap-ab-a100-2g-pq-r1 | skip_existing_valid |
| a100-2g | 2 | - | - | pq | logs/awave-cw2-overlap-ab-a100-2g-pq-r2 | skip_existing_valid |
| a100-2g | 3 | - | - | pq | logs/awave-cw2-overlap-ab-a100-2g-pq-r3 | skip_existing_valid |
| a100-2g | 4 | awave-cw2-overlap-ab-a100-2g-pq-r4-mvnxf | awave-cw2-overlap-ab-a100-2g-uq-r4-ntm5n | pq | logs/awave-cw2-overlap-ab-a100-2g-pq-r4 | deleted_loser_job=awave-cw2-overlap-ab-a100-2g-uq-r4-ntm5n |
| a100-2g | 5 | awave-cw2-overlap-ab-a100-2g-pq-r5-hwdlb | awave-cw2-overlap-ab-a100-2g-uq-r5-kbbjj | pq | logs/awave-cw2-overlap-ab-a100-2g-pq-r5 | deleted_loser_job=awave-cw2-overlap-ab-a100-2g-uq-r5-kbbjj |
| h100-2g | 1 | - | - | pq | logs/awave-cw2-overlap-ab-h100-2g-pq-r1 | skip_existing_valid |
| h100-2g | 2 | - | - | pq | logs/awave-cw2-overlap-ab-h100-2g-pq-r2 | skip_existing_valid |
| h100-2g | 3 | - | - | pq | logs/awave-cw2-overlap-ab-h100-2g-pq-r3 | skip_existing_valid |
| h100-2g | 4 | awave-cw2-overlap-ab-h100-2g-pq-r4-9kbsj | awave-cw2-overlap-ab-h100-2g-uq-r4-j2x4k | pq | logs/awave-cw2-overlap-ab-h100-2g-pq-r4 | deleted_loser_job=awave-cw2-overlap-ab-h100-2g-uq-r4-j2x4k |
| h100-2g | 5 | awave-cw2-overlap-ab-h100-2g-pq-r5-sjnvd | awave-cw2-overlap-ab-h100-2g-uq-r5-8gl5d | uq | logs/awave-cw2-overlap-ab-h100-2g-uq-r5 | deleted_loser_job=awave-cw2-overlap-ab-h100-2g-pq-r5-sjnvd |
