# Changelog

## [1.6.0] - 2026-04-28

### JQuamtsScreeningBot（MA200局面・candidate_lane・データレビュー）

- **200日線局面分類**（`evaluate_ma200_entry_state`）：reclaim / basing / extended / downtrend 等。`recent_60d_low_update`（直近N日での60日底更新）をレーン判定に利用。
- **`fundamental_edge_score`**（0–100）と **`compute_entry_score`** による生エントリー得点を分離。表示用 **`entry_score`** は **`cap_entry_score`** でレーン別上限（`LANE_ENTRY_CAP` / `LANE_CAP_*` 環境変数）。
- **`assign_entry_candidate_lane`**：`ma200_reclaim_core`、`bottom_reversal_core`、`weak_reclaim_watch`、`watch_fundamental_core`、`extended_above_ma200`、`data_review` / `data_review_light`、`cyclical_value_trap` 等。reclaim/core は Piotroski（生>3、調整≥6、カバレッジ≥0.6）、PEG警告、営業利益安定を要件化。bottom はファンダ≥75・basing・調整≥6.5・21日リターン正・60日底未更新など。
- **`compute_data_review_meta`**：`data_review_reason` / `data_review_level`（light/medium/severe）。フラットCSV・投資助言MD・トップMDに出力。
- **営業利益急落判定**：`OP_INCOME_DROP_FLOOR` デフォルト **0.6**（環境変数で上書き可）。レポート必須フィルタに明示。
- **セクター**：`Sector33Name` 優先、`normalize_sector` / `SECTOR_MEDIANS` 拡張で「その他」依存を低減。
- **推奨並び**：`_build_ranked` で **`rec_priority`** / **`rec_secondary`**（watch はファンダ優先）。`top_recommended_*.csv` と助言レポートで利用。MD表は `legacy_total` / `fundamental_edge` / `entry_timing_score` 表記と注記を追加。
- **ドキュメント**：`docs/スクリーニング仕様マニュアル.md` §7 に本仕様を記載。

タグ: `v1.6.0`

## [1.5.1] - 2026-04-28

### JQuamtsScreeningBot（収集フィルタ・マスタ正規化・CF 別名）

- Normalized V2 `equities/master` using `CoName`, `CoNameEn`, `S33Nm`, `Mkt`, `MktNm` (fixes empty `CompanyName` in cached lists).
- Strengthened `filter_collectable_equities_df`: ETF segment (`Mkt=0109`), keyword exclusions, empty-name exclusion for batch flows, JP code-band fallbacks (`1300–1399`, `1450–1499`, `1550–1699`) when not prime-listed; single `--phase single` bypasses this filter.
- One-line INFO log with removal counts; new daily cache file `sector_stock_list_v2norm_{date}.csv`.
- Map V2 summary `CFO` into `NetCashProvidedByUsedInOperatingActivities` and `CashFlowsFromOperatingActivities` in aliases and legacy conversion.
- Default collect batch size via `JQ_DEFAULT_BUDGET` (unset → 800) for CLI and menus.
- Resolve `api.ini` beside the script directory; read `[DEFAULT] API_KEY` without relying on `has_section("DEFAULT")`.

タグ（収集フィルタ初版）: `v1.5.1-jquants-v2-collect-filter`

### J-Quants V2 instrument split snapshot

- Added `instrument_type` classification for stock / etf / etn / reit / fund / fund_like / unknown.
- Separated ETF/ETN/REIT/fund_like instruments from individual-stock financial screening.
- Preserved ETF-like instruments in an unscored ETF candidate lane (`output/reports/etf_candidates_unscored.csv`, gitignored) for future ETF-specific screening.
- Ensured Mkt/MktNm-based stock classification (`0105`, `0111`–`0113`, and MktNm hints) takes precedence over fallback code-band filtering, preventing false exclusion of ordinary stocks (e.g. 2002 band).
- Added/propagated `instrument_type` to analysis and flattened CSV output.
- Reduced noisy per-symbol legacy-statement warnings by aggregating missing financial statements for stocks.
- Improved CFO alias handling for V2 summary-only financial conversion.
- ETF scoring remains out of scope; future ETF scoring should use AUM, expense ratio, NAV deviation, liquidity, and trend/pullback metrics (and external data where J-Quants does not provide fields).

タグ（instrument split）: `v1.5.1-jquants-v2-instrument-split`

## [1.5.0] - 2026-04-28

### JQuamtsScreeningBot（J-Quants API V2 / summary-only 互換）

- Migrated J-Quants integration to API V2 using x-api-key authentication.
- Added V2 cache separation under `.jquants_cache_v2`.
- Added pagination support for V2 endpoints.
- Added V2 price normalization for equities/bars/daily.
- Added V2 summary-only compatible financial conversion.
- Added `financial_data_mode` / `fins_details_*` diagnostics.
- Added handling for fins/details 403 by disabling repeated detail requests in the same session.
- Added `api.ini.example` (committed template). Copy to local-only `api.ini` for `[DEFAULT] API_KEY`; `api.ini` stays gitignored.

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
