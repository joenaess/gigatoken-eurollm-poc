## Benchmark summary

Tokenizer: `openeurollm/tokenizer-256k-v2` · Corpus: 50.0 MB, 23,908 documents, 10,346,407 tokens · Warmup: 1 · Timed iterations: 3 (best time reported)

Hardware: 13th Gen Intel(R) Core(TM) i9-13980HX (24 cores / 32 threads, 31.0 GB RAM) · Python 3.12.3 · gigatoken 0.10.0, tokenizers 0.23.3, transformers 5.19.0, sentencepiece 0.2.2, numpy 2.5.4

### Overall

| Engine                          | Best (s)   | Mean ± std (s)   | MB/s   | Mtok/s   | Speedup   |
|---------------------------------|------------|------------------|--------|----------|-----------|
| HF Fast (encode_batch)          | 0.448      | 0.456 ± 0.008    | 111.6  | 23.08    | 1.00×     |
| HF Fast (encode_batch_fast)     | 0.378      | 0.386 ± 0.007    | 132.2  | 27.35    | 1.19×     |
| Gigatoken HF compat             | 0.165      | 0.171 ± 0.008    | 302.2  | 62.52    | 2.71×     |
| Gigatoken native (encode_files) | 0.050      | 0.052 ± 0.002    | 1003.9 | 207.70   | 9.00×     |

Speedup is relative to **HF Fast (encode_batch)** (best-time ratio).

### Per domain — MB/s (speedup)

| Engine                          | prose          | code           | chat          | math          |
|---------------------------------|----------------|----------------|---------------|---------------|
| HF Fast (encode_batch)          | 117.8 (1.00×)  | 59.1 (1.00×)   | 104.6 (1.00×) | 80.8 (1.00×)  |
| HF Fast (encode_batch_fast)     | 150.1 (1.27×)  | 74.7 (1.26×)   | 135.0 (1.29×) | 112.2 (1.39×) |
| Gigatoken HF compat             | 354.1 (3.01×)  | 299.0 (5.06×)  | 370.6 (3.54×) | 105.9 (1.31×) |
| Gigatoken native (encode_files) | 1114.3 (9.46×) | 626.9 (10.61×) | 616.3 (5.89×) | 157.7 (1.95×) |
