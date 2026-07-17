# Expanded-649 LDA-only Full ACO Latency

- Accuracy replay: 120/128 = 93.75%; adopted-mainline mismatch count = 0.
- Feature-ready latency: median 7.652 ms, P95 8.093 ms.
- Packet-to-location latency: median 15.849 ms, P95 18.122 ms.

The online path includes RSSI+, S17, q1/q4 extraction, LDA posterior, alpha candidate fusion, four-segment ACO/Score4, and beta final fusion. RF is not loaded into inference. Over-the-air reception, synchronization, model/template preload, and disk output are excluded.
