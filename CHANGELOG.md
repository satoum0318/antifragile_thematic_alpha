# Changelog

## [1.5.0] - 2026-04-28

### JQuamtsScreeningBot（J-Quants API V2 / summary-only 互換）

- Migrated J-Quants integration to API V2 using x-api-key authentication.
- Added V2 cache separation under `.jquants_cache_v2`.
- Added pagination support for V2 endpoints.
- Added V2 price normalization for equities/bars/daily.
- Added V2 summary-only compatible financial conversion.
- Added `financial_data_mode` / `fins_details_*` diagnostics.
- Added handling for fins/details 403 by disabling repeated detail requests in the same session.
- Kept `--budget` default at 380; larger V2 collection should be tested gradually.

## [1.4.0] - 2026-04-14

### JQuamtsScreeningBot（スクリーニング安定化）

- **EPS 成長率**: 当期・前期で**別々の株式数**を用いて定義。前期株数欠損時は計算しない。
- **バリュエーション**: PS / PER / PEG 入力の truthy 判定をやめ、`is not None` / `> 0` など明示化。
- **Piotroski**: 欠損を 0 とみなさない三値論理（True / False / None）とカバレッジメタデータ。
- **年次以外の決算系列**: `sales_cagr` と営業利益安定は**非計算**（annual 以外では混在回避）。
- **安全性 / 投機性**: `eps_growth_rate` を `calculate_safety_score_v3` / `detect_speculative_manipulation_v2` に接続。
- **CLI**: Windows cp932 等でも `--phase analyze` が落ちないよう `_cli_print` で表示を安全化。
- **レート制限**: `AuthSession.request` は `super().request` が返るたび `mark()`（認証 POST は limiter 外と明記）。
- **診断**: `valuation_input_complete` に前期株数条件、`eps_growth_input_complete` を追加。`growth_proxy_detached_from_scoring` は EPS が safety/spec に接続されたため **False** に更新。

タグ: `v1.4.0-jquants-screening-stability`
