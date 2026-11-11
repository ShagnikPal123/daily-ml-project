### T1. Row-level extraction F1 vs OCR noise level (mean ± 95% CI over seeds)

| method | noise 0.0 | noise 0.01 | noise 0.02 | noise 0.04 | noise 0.08 |
|---|---|---|---|---|---|
| naive-regex | 0.110 ± 0.043 | 0.118 ± 0.022 | 0.101 ± 0.023 | 0.091 ± 0.023 | 0.101 ± 0.053 |
| rules | 1.000 ± 0.000 | 0.931 ± 0.012 | 0.881 ± 0.020 | 0.749 ± 0.025 | 0.571 ± 0.102 |
| rules+ocr-repair | 1.000 ± 0.000 | 0.941 ± 0.009 | 0.905 ± 0.018 | 0.795 ± 0.026 | 0.644 ± 0.090 |
| tagger (ours) | 0.983 ± 0.023 | 0.971 ± 0.015 | 0.975 ± 0.010 | 0.954 ± 0.016 | 0.891 ± 0.027 |

### T2. Tagger ablations (noise 0.02)

| variant | row F1 | token accuracy |
|---|---|---|
| full | 0.977 ± 0.015 | 0.997 ± 0.002 |
| no Viterbi (independent argmax) | 0.964 ± 0.006 | 0.995 ± 0.001 |
| no context features | 0.754 ± 0.078 | 0.979 ± 0.008 |
| no position features | 0.881 ± 0.013 | 0.988 ± 0.002 |
| no shape features | 0.964 ± 0.019 | 0.995 ± 0.002 |
| no lexical features | 0.972 ± 0.016 | 0.996 ± 0.002 |
| no OCR repair in assembly | 0.948 ± 0.023 | 0.997 ± 0.002 |

### T3. Layout shift (noise 0.02)

| condition | row F1 |
|---|---|
| tagger trained on all layouts, tested on inline | 0.946 ± 0.043 |
| tagger trained on pipe+space only, tested on inline | 0.000 ± 0.000 |
| tagger trained on pipe+space only, tested on pipe+space | 0.983 ± 0.012 |
| rules (parsers cover all layouts), tested on inline | 1.000 ± 0.000 |

### T4. Retrieval, header=True|para=0.0

| method | recall@1 | recall@5 | MRR |
|---|---|---|---|
| tfidf | 0.344 ± 0.050 | 0.892 ± 0.047 | 0.554 ± 0.037 |
| bm25 | 0.461 ± 0.063 | 0.949 ± 0.034 | 0.670 ± 0.053 |
| lsa | 0.585 ± 0.049 | 0.948 ± 0.028 | 0.732 ± 0.031 |
| bm25+thesaurus | 0.429 ± 0.048 | 0.945 ± 0.036 | 0.637 ± 0.044 |
| bm25+part-filter | 0.588 ± 0.037 | 0.997 ± 0.007 | 0.778 ± 0.024 |
| hybrid(bm25,lsa)+thesaurus | 0.521 ± 0.081 | 0.949 ± 0.027 | 0.695 ± 0.057 |
| hybrid+thesaurus+part-filter | 0.636 ± 0.050 | 1.000 ± 0.000 | 0.806 ± 0.028 |
| random | 0.005 ± 0.007 | 0.015 ± 0.014 | 0.026 ± 0.009 |

### T4. Retrieval, header=True|para=0.5

| method | recall@1 | recall@5 | MRR |
|---|---|---|---|
| tfidf | 0.252 ± 0.034 | 0.753 ± 0.040 | 0.452 ± 0.030 |
| bm25 | 0.320 ± 0.045 | 0.816 ± 0.030 | 0.537 ± 0.038 |
| lsa | 0.460 ± 0.026 | 0.849 ± 0.017 | 0.622 ± 0.018 |
| bm25+thesaurus | 0.356 ± 0.049 | 0.904 ± 0.035 | 0.585 ± 0.040 |
| bm25+part-filter | 0.464 ± 0.048 | 0.967 ± 0.006 | 0.683 ± 0.028 |
| hybrid(bm25,lsa)+thesaurus | 0.443 ± 0.060 | 0.907 ± 0.037 | 0.634 ± 0.044 |
| hybrid+thesaurus+part-filter | 0.577 ± 0.056 | 0.999 ± 0.004 | 0.771 ± 0.033 |
| random | 0.004 ± 0.007 | 0.024 ± 0.011 | 0.029 ± 0.009 |

### T4. Retrieval, header=False|para=0.5

| method | recall@1 | recall@5 | MRR |
|---|---|---|---|
| tfidf | 0.085 ± 0.018 | 0.467 ± 0.052 | 0.245 ± 0.022 |
| bm25 | 0.085 ± 0.016 | 0.563 ± 0.053 | 0.274 ± 0.019 |
| lsa | 0.135 ± 0.014 | 0.548 ± 0.045 | 0.305 ± 0.026 |
| bm25+thesaurus | 0.088 ± 0.021 | 0.611 ± 0.045 | 0.297 ± 0.021 |
| bm25+part-filter | 0.309 ± 0.048 | 0.968 ± 0.014 | 0.599 ± 0.034 |
| hybrid(bm25,lsa)+thesaurus | 0.129 ± 0.025 | 0.607 ± 0.017 | 0.318 ± 0.025 |
| hybrid+thesaurus+part-filter | 0.621 ± 0.039 | 0.999 ± 0.004 | 0.793 ± 0.025 |
| random | 0.004 ± 0.007 | 0.024 ± 0.011 | 0.029 ± 0.009 |

### T5. End-to-end QA exact-match accuracy vs noise

| pipeline | noise 0.0 | noise 0.02 | noise 0.04 | noise 0.08 |
|---|---|---|---|---|
| bm25 + naive regex | 0.105 ± 0.068 | 0.119 ± 0.052 | 0.104 ± 0.040 | 0.105 ± 0.060 |
| bm25 + rules | 0.751 ± 0.061 | 0.571 ± 0.034 | 0.481 ± 0.058 | 0.347 ± 0.078 |
| hybrid-routed + rules | 0.972 ± 0.020 | 0.769 ± 0.053 | 0.625 ± 0.086 | 0.447 ± 0.086 |
| hybrid-routed + rules+repair | 0.972 ± 0.020 | 0.805 ± 0.030 | 0.681 ± 0.083 | 0.539 ± 0.096 |
| hybrid-routed + tagger (ours) | 0.951 ± 0.033 | 0.932 ± 0.047 | 0.911 ± 0.028 | 0.859 ± 0.046 |

### T6. Outcome breakdown for the headline pipeline (counts pooled over seeds)

| noise | correct | retrieval_miss | extract_error | wrong_row |
|---|---|---|---|---|
| 0.0 | 713 | 1 | 16 | 20 |
| 0.02 | 699 | 0 | 29 | 22 |
| 0.04 | 683 | 1 | 44 | 22 |
| 0.08 | 644 | 0 | 92 | 14 |

### T7. LSA dimension sweep, recall@5 (paraphrase 0.5)

| method | k=8 | k=16 | k=32 | k=64 | k=128 |
|---|---|---|---|---|---|
| lsa | 0.152 ± 0.034 | 0.496 ± 0.048 | 0.719 ± 0.052 | 0.849 ± 0.030 | 0.777 ± 0.023 |
| hybrid+thesaurus+part-filter | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.997 ± 0.005 |

### T8. Chunk-size sweep

| lines/chunk | #chunks | recall@1 | recall@5 | end-to-end acc |
|---|---|---|---|---|
| 2 | 330 | 0.675 ± 0.021 | 1.000 ± 0.000 | 0.963 ± 0.032 |
| 4 | 205 | 0.593 ± 0.043 | 1.000 ± 0.000 | 0.955 ± 0.036 |
| 8 | 166 | 0.420 ± 0.040 | 0.985 ± 0.009 | 0.940 ± 0.029 |

### T9. Index compression of LSA embeddings (k=64)

| bits/dim | index bytes | recall@1 | recall@5 | MRR |
|---|---|---|---|---|
| 32 | 52429 | 0.513 ± 0.055 | 0.911 ± 0.021 | 0.675 ± 0.031 |
| 8 | 13926 | 0.516 ± 0.056 | 0.911 ± 0.021 | 0.676 ± 0.031 |
| 4 | 7373 | 0.513 ± 0.063 | 0.909 ± 0.030 | 0.678 ± 0.035 |
| 2 | 4096 | 0.247 ± 0.073 | 0.725 ± 0.020 | 0.424 ± 0.047 |

BM25 sparse index (float32 values + int32 indices): 28570 bytes on average.

### T10. Fixed-point tagger (noise 0.02): row F1 and model size

| weights | row F1 | dense bytes | sparse bytes |
|---|---|---|---|
| float64 reference | 0.959 ± 0.025 | 1179972 | 1179972 |
| int8 | 0.960 ± 0.025 | 295272 | 11453 |
| int6 | 0.960 ± 0.023 | 221544 | 10606 |
| int4 | 0.956 ± 0.021 | 147816 | 5498 |
| int3 | 0.895 ± 0.058 | 110952 | 2509 |
| int2 | 0.367 ± 0.263 | 74088 | 644 |

### T11. Profiling (seed 0, 206 chunks; machine dependent)

| component | cost |
|---|---|
| naive-regex | 7 us/line |
| rules | 5 us/line |
| tagger float | 354 us/line |
| tagger int8 | 260 us/line |
| bm25 | 22 us/query |
| lsa | 47 us/query |
| hybrid+thesaurus+part-filter | 322 us/query |

### T12. Tagger token confusion at noise 0.08 (token accuracy 0.9883)

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| O | 0.998 | 1.000 | 0.999 | 14077 |
| NAME | 0.997 | 0.995 | 0.996 | 3011 |
| SYM | 0.997 | 0.984 | 0.991 | 1205 |
| COND | 0.961 | 0.969 | 0.965 | 1293 |
| MIN | 0.903 | 0.872 | 0.887 | 917 |
| TYP | 0.927 | 0.882 | 0.904 | 1029 |
| MAX | 0.975 | 0.979 | 0.977 | 1047 |
| UNIT | 0.998 | 0.996 | 0.997 | 1395 |
| SEP | 0.990 | 0.999 | 0.994 | 5500 |

Largest confusions (gold -> predicted: count): TYP->MIN: 73, MIN->TYP: 57, MIN->COND: 33, MIN->SEP: 22, TYP->MAX: 22, COND->SEP: 21
