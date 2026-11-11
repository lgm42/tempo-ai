# Changelog

## 0.1.0 — 2026-11-11

First version.

- Forecast of the Tempo color at D+2, D+3 and D+4 (and D+1 before RTE publishes it), with probabilities.
- Daily self-evaluation: forecasts are stored and compared with the real colors.
- REST API (`/api/summary` gives everything in one request) and a small web page.
- Automatic data collection and retraining (every 30 days by default).
- Unit tests (rules, statistics, forecast, API) run by CI before each image build.
- 2026-27 red days: approximated with the 2014 algorithm within the CRE monthly range, as RTE has not
  published the PP thresholds yet.
