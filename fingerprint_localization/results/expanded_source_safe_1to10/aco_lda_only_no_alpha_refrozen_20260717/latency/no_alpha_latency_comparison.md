# No-alpha refrozen latency comparison

Both variants were benchmarked back-to-back in the same process.

| Boundary | alpha=0.3 median/P95 | no-alpha median/P95 | median delta |
| --- | ---: | ---: | ---: |
| Feature-ready | 7.605/8.042 ms | 7.476/7.896 ms | -1.70% |
| Packet-to-location | 17.189/20.399 ms | 17.156/20.489 ms | -0.19% |

Candidate ranking median changed from 1.365 to 1.368 ms (+0.24%).
