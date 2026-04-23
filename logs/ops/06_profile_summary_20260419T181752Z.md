# 06 Profile Summary (2026-04-19 18:17:53 UTC)

- processed_dirs: 10
- skipped_dirs: 7 (pending/incomplete treated as skipped)

| Source Dir | GPU | nGPU | Size | interior(ms) | waitall_avg(ms) | comm_frac(%) | overlap_proxy(%) |
|---|---|---:|---:|---:|---:|---:|---:|
| nsys_scale_a100_np2_uq_r2 | A100 | 2 | 1000 | 13.5638 | 0.242087 | 3.20955 | 96.4304 |
| nsys_scale_a100_np4_pq | A100 | 4 | 1000 | 6.77025 | 0.703169 | 16.1145 | 79.2277 |
| nsys_scale_a100_np4_uq | A100 | 4 | 1000 | 6.77193 | 0.563353 | 13.3351 | 83.3621 |
| nsys_scale_h100_np2_pq_r2 | H100 | 2 | 1000 | 7.15719 | 0.66765 | 14.8189 | 81.3432 |
| nsys_scale_h100_np2_uq_r2 | H100 | 2 | 1000 | 7.16608 | 0.470152 | 10.8988 | 86.8784 |
| nsys_scale_h100_np4_pq | H100 | 4 | 1000 | 3.51332 | 1.41741 | 42.8135 | 19.3122 |
| nsys_scale_h100_np4_pq_r5 | H100 | 4 | 1000 | 3.51287 | 6.31357 | 76.9335 | 0 |
| nsys_scale_h100_np4_uq_r5 | H100 | 4 | 1000 | 3.51324 | 1.28184 | 40.3726 | 27.028 |
| nsys_scale_h100_np8_pq | H100 | 8 | 1000 | 1.75346 | 2.88923 | 75.2902 | 0 |

## Skipped
- nsys_roofline_h100: missing_csv_pair
- nsys_scale_a100_np2_pq: missing_csv_pair
- nsys_scale_a100_np2_uq: missing_csv_pair
- nsys_scale_h100_np2_pq: missing_csv_pair
- nsys_scale_h100_np2_uq: missing_csv_pair
- nsys_scale_h100_np4_pq_r3: missing_csv_pair
- nsys_scale_h100_np4_uq_r3: missing_csv_pair
