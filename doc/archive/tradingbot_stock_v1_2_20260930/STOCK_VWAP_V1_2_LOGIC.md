# 現物株・現物ETF 自動売買ロジック（stock_vwap_v1_2）

**route id**: `stock_vwap_v1_2`  
**実装**: `TradingBot` + `stock_vwap_recovery_v12`（純関数・定数）
**last_verified**: 2026-09-30（現行コードを正として再確認）
**release**: app `v3.3.251` / GUI `v1.176` / `stock-vwap-v1.2-code-authority-20260930`

---

## スコープ

- **対象**: `TradingBot` の現物ユニバース処理と `src/trading/stock_vwap_recovery_v12.py` のみ。
- **J-Quants** の銘柄スクリーニングや別プロジェクトのロジックとは**無関係**。
- **Piotroski / PER / 財務条件**は本ルートに**含まれない**（テクニカルのみ）。

---

## 現在の有効ユニバース

**ETF**（`ETF_SETTINGS` → `ALL_STOCK_SETTINGS`）:

- **1321**（`min_unit=1`, `max_position=3`）
- **1540**（`max_position=10`）

**個別株**（`STOCK_SETTINGS`）:

- **3048, 7752, 6758, 8098, 8306, 4114, 6995, 4493, 7287, 3549**

**対象外・別経路**

- **1570**: IBKR 口座で Complex/Leveraged ETP 権限が要る場合があり、現行N225 ETF枠は1321へ寄せる。
- **7419**: 過去に追加されたが現行ユニバースから削除済み。実効正本は `trading_config.STOCK_SETTINGS`。
- **2644**: 現運用ではユニバース外。将来 `ETF_SETTINGS` に戻せば v1.2 候補として動作確認済み（`trading_config` メモ）。
- **仮想ETFデイトレ**（旧 `ETFIntegration` 等）: legacy cleanup で物理削除済み。現物ETFスイングとは別経路。
- **family lock**: コードには入れず**運用で管理**。

---

## 日足ロジック

- **確定日足のみ**（`_confirmed_rows_excluding_today`）。当日未確定足は使わない。
- **raw_long** = `(close > SMA50) and (close > SMA200)`（各確定行で評価）。
- **raw_exit** = `(close < SMA50)`。
- **daily_direction** は **`LONG` / `NEUTRAL` のみ**（`None` は使わない。正規化は `TradingBot._v12_daily_dir_norm`）。
- **LONG へ遷移**: `raw_long` が**2 営業日連続**で真。
- **NEUTRAL へ遷移**: `raw_exit` が**2 営業日連続**で真。
- **それ以外**: 前状態を維持（LONG か NEUTRAL のまま）。
- **データ不足**（`evaluate_daily_regime_v12` が `ok_data=False`）: **`daily_direction = NEUTRAL`**、**`_daily_exit_flag = False`**（stale LONG で新規させない）。
- ヒステリシス計算には直近 2 日分の SMA が有効である必要があるため、確定行は **`MIN_DAILY_REGIME_ROWS`（201）** 本以上を要する（`MIN_HIST_DAYS=200` に加え +1）。

**実装**: `evaluate_daily_regime_v12`、純関数 `next_daily_direction_hysteresis` / `row_raw_flags` / `daily_exit_flag_from_raw_exit`。  
**TradingBot**: `_set_daily_from_dailydf_stock_v12` で `prev_direction` を渡し、結果を `daily_direction` / `_daily_exit_flag` に反映。

---

## イントラ新規（執行エンジン）

- **5 分確定足**のみ（`_last_closed_bar_index`）。**日足ロジックは変更なし**。
- **`daily_direction == LONG`**（エントリ評価では **`reg.ok_data` かつ `reg.direction == LONG`** を参照し、保存済み LONG だけに依存しない）。
- **寄り付き直後マスク** `[SESSION_VWAP_START, OPEN_AUCTION_MASK_END)` = **09:00–09:05 JST** → **`C_OPEN_AUCTION_MASK`**（hard block）。**`ENTRY_NOT_BEFORE`（09:30）未満**は **`C_BEFORE_0930`**（hard block）。
- **セッション VWAP** は **JST営業日ごと・09:00起点**で日次リセットする。実運用callerが30日分5分足を渡しても日跨ぎで累積しない。09:00前の行は前営業日のVWAPで `ffill` しない。
- **VWAP touch / reclaim / 陽線（bullish 5m）は hard gate ではない**。`soft_observations` に載る観測のみ。**日足 LONG の買い候補を潰さない**。
- **時間帯別の観測用執行プラン**（確定足の `close`・セッション VWAP・ATR5）。ここで算出する `chosen_limit_price` はログ・高値追い判定用であり、通常BUYの実発注価格には使わない:
  - **09:30–11:00** `morning`: `min(close, vwap)`
  - **11:00–14:00** `mid`: `min(close, vwap + 0.2×ATR)`（ATR 欠損時は終値の 1% などのフォールバック）
  - **14:00–14:50** `late`: `min(close+1tick, vwap + 0.5×ATR)`
  - **14:50 以降** `close_rush`: **未約定の買い指値があればキャンセル**のうえ **成行買い**（`buy_market_immediately`）。**ログ** `fallback_to_market=1`。
- **morning / mid / late の実発注**: `process_order` は上記 `chosen_limit_price` を通常v1.2 BUYの価格overrideには使わず、最新 ask の **marketable LMT → 残数量 MKT fallback** へ進む。板の spread tick と ask 表示数量カバレッジに応じて LMT 待機を `fast=5s` / `normal=15s` / `slow=30s` に分類し、未約定残は LMT cancel terminal 確認・建玉差分・pending BUY・`max_position` 再照合後に MKT で取り切る。薄板・広spreadでも価格条件だけではMKTを断念しないが、ask欠損・取消未確認・建玉不明・snapshot不信頼は hard block。
- **約定後のストップ**: 通常BUYは `process_order` 内の約定後確認で `pending_stop` → `active_stop` へ昇格する。外部/遅延反映の建玉は引き続き **`process_stock` で IB 建玉 > 0 を見て** `pending_stop` → `active_stop` へ昇格する。
- **外部/手動建玉の保護ストップ**: `active_stop` が空のまま IB 建玉だけ存在する場合、`process_stock` が最新 5 分足と `entry_price`（不明時は最新 close）から **retrofit `stop_line`** を生成して `active_stop` を自動 Arm する（`[STOCK_V12] stop_armed_from_ib_position`）。
- **高値追い一時見送り**: 確定 `close > vwap + 1.5×ATR` かつ **close_rush 以外**は **`C_OVEREXTENDED_DEFER`**（当該サイクルは注文しない）。**引け前フェーズでは無効**（日足 LONG を優先して買いに行く）。
- **GU（寄りギャップ）ガードは使用しない**（削除済み）。

**実装**: `evaluate_intraday_entry_v12(..., daily_long_ok=True, atr5=...)`。**観測ログ**: **`[STOCK_V12_EXEC]`**。マスク中は TradingBot が **`[ENTRY-MASK]`**（同一銘柄 300 秒に 1 回まで）。

---

## 新規買い（発注ゲート）

戦略条件（日足 LONG・イントラ `intr.ok`・余力 `cash_ok` 等）を満たしたうえで **`process_order(BUY)`** を呼ぶ。**共通ゲートは `process_order` 内で一度だけ**かける（呼び出し側での二重 `check_trading_conditions` はしない）。

**責務の積み上げ**（現物 v1.2）:

1. **判定**: 日足 `reg`、イントラ **`evaluate_intraday_entry_v12`（執行計画・hard はデータ・マスク・過伸展見送りのみ）**、`_evaluate_new_entry` 内の `cash_ok`。
2. **初回数量**: `trading_config.py` の銘柄別 **`max_position` までの残枠**を要求数量とし、`min_unit` 単位へ切り下げたうえで `AvailableFunds`（約定価格変動用 1% バッファ込み）で買える数量へクリップする。1321 は現在の設定で最大3枚、1540は10枚、個別株は各100株。設定変更時も固定値ではなく `max_position` に追従する。真に `AvailableFunds` が不明なら `account_available_funds_unavailable` で fail-closed。
3. **再エントリー予算**: `NetLiquidation` 不明かつ当日STOP損失0円なら `0 >= 0` とみなさず、日次損失予算だけをスキップする。当日STOP損失が正で `NetLiquidation` 不明なら `netliq_unavailable_for_loss_budget` で fail-closed。予算超過・回数上限・cooldown・価格回復不足は個別 reason を返す。
4. **共通ゲート**: **`check_trading_conditions`**（IB・`is_market_open`・`synchronize_positions`・SELL 時のキャッシュ枚数整合・**BUY のみ** 300 秒ルール）。open order 観測は `openTrades` だけに依存せず、`reqAllOpenOrders` / `openOrdersAsync` を併用して可視性を補強。戻り値 **`(allowed, reason)`**。ブロック時は **`[ORDER-GUARD]`** および **`[ORDER] check_trading_conditions blocked ... reason=...`**（WARNING）。
5. **最終防衛**: **`_validate_order_conditions`**（銘柄設定の存在・**最小枚数**のみ。市場二重チェックはしない）。
6. **執行**: 主経路のBUY上限は **`process_order` / `_revalidate_order_quantity_before_place`** の対象account+conId strict snapshotで計算する。legacy `clip_buy_qty_to_limits_async` は補助・互換用であり正本ではない。v1.2 通常BUYは **`execute_dynamic_order_with_fallback(stock_v12_marketable_entry=True)`** に集約し、最新 ask の marketable LMT から始め、流動性連動待機後の未約定残だけ MKT にする。**close_rush** は従来どおり `buy_market_immediately=True` の即時MKT。
7. **上限超過ポジション保護**: reconcile 時に `ib_pos > max_position` を検知した場合は **新規 BUY を当日ブロック**し、リスク拡大を停止する（SELL は継続許可、`[RISK_LIMIT_BREACH]` ログ）。
8. **P64 cache sync**: 起動・再接続・通常cycle開始時は、`generation`/鮮度/`trusted`付きのIB snapshotで同期してから評価する。注文が`placeOrder`へ到達した場合はcycle snapshotをdirty化し、次銘柄の前に再取得する。取得失敗・account/conId曖昧時はローカルstateを上書きせず、BUY blockを維持する。
9. **active stop中の追加BUY**: `active_stop`がある銘柄は、追加BUY禁止を維持したままentry評価の上流で除外する。初回注文の未約定残は同一注文の LMT→MKT fallback で取り切り、その後の別サイクルでは増し玉しない。

**`process_order` 入口**の **`[ORDER] enter ...`** で「関数到達」を切る。

---

## Exit（日足 `_daily_exit_flag`）

- **raw_exit が 2 営業日連続**でレジーム上は **NEUTRAL 遷移**と整合（`_daily_exit_flag` の価格条件と同型）。
- **日足ホールドを外れたら `_daily_exit_flag = True`**。**個別株・現物 ETF ともに**、**`position > 0` の間は日足 exit の売り実行対象**。
- **個別株を「寄り窓にしか売らない」旧仕様には戻さない**。`_try_daily_exit_sell` が **`process_stock`** / **`_flush_pending_daily_exits`** から呼ばれ、**フラットまで exit intent を維持**する。
- **寄り付き直後マスク（09:00–09:05）**中は **daily exit の発注だけ保留**（`_daily_exit_flag` は維持）。マスク解除後、`position > 0` かつ pending sell なしなら **従来どおり再 SELL**。**観測**: **`[DAILY-EXIT-MASK]`**（300 秒に 1 回まで）。
- **未約定の売り（pending sell）がある間は二重売りしない**（`[DAILY-EXIT-PENDING]`）。**pending が消えても建玉が残っていれば次サイクルで再 SELL**（`[DAILY-EXIT-RETRY]`）。
- **P64**: 日足exit SELL前の建玉・pending sell確認も、対象account・conId一致のIB snapshotを正とする。identity曖昧時はSELLを出さず、BUY blockのみ立てて状態上書きを避ける。
- **ストップ安・特売り・未約定・失効・部分約定**で一度売れなくても、**フラットになるまで意図を維持**（フラグは勝手に落とさない）。
- **P67**: STOP / 日足EXITのSELL直前に、対象account+conIdのworking BUYがあればcancelし、terminal ACKまたはremaining=0をtrustedに確認してからSELLする。確認不能ならSELLを出さずexit intentを保持する。
- **`_daily_exit_flag` をクリアするのは**、`process_order(SELL)` 後の同期で **対象account+conIdの`pos_after == 0`かつ`pending_sell == 0`** を確認したときだけ（`[DAILY-EXIT-CLEAR]`）。partial fill、TWS反映遅延、pending SELL残存時はフラグ維持。理由文字列に **`Daily exit`** が含まれる SELL のみ intent 解除対象（フォールバック売りは別理由）。
- **日足系の売り口**は **`_try_daily_exit_sell`** に集約（`process_stock`、flush、**`daily_dir==EXIT` かつ flag 不整合のフォールバック**）。マスク・pending・episode ログを共有する。
- **場中ストップ（現物 TradingBot v1.2 のみ・先物デイトレとは無関係）**:
  - **現物ユニバース全体**（個別株・現物 ETF）: 設定プロファイルは **`swing`**（`stop_version=swing_v1`）。距離は `max(2tick, entry_price×1.5%, entry_bar_range×0.25, ATR実効値×1.0)`、anchorは `min(entry_price, entry_bar_low)`。**主出口は日足 exit**。
  - **現行のATR実効値**: `_stock_v12_atr_for_stop_arm` は日足ATRを最優先するが、公式日足更新 `_set_daily_from_dailydf_stock_v12` が `daily_data` をOHLCV＋SMA50/SMA200へ正規化して保存するため、通常経路では `daily_data["atr"]` が残らない。この場合、初回BUYはエントリー位置の5分ATR → イントラ評価ATR、外部/手動建玉のretrofitは最新確定5分ATRへフォールバックする。いずれも無ければ swing の `max(entry_price×1.5%, 2tick)` をATR実効値とする。したがって現行コードを「常に日足ATR×1.0」とは扱わない。
  - レガシー **`intraday`**（5 分 ATR × 0.35 + 0.3%）は `STOCK_V12_STOP_PARAMS_BY_PROFILE` に参照用として残存。現行運用では未使用。
  - 設定: `trading_config.STOCK_V12_STOP_PROFILE_BY_PRODUCT` / `STOCK_V12_STOP_PARAMS_BY_PROFILE`。
  - 発火: **確定 5 分足 close** が `stop_line` 未満（`closed_bar_close_for_stop`）。形成中足は使わない。
  - **09:00–09:05 JST は STOP 発注を保留**し、`active_stop` は維持する。昼休みは `is_market_open=False` のため、11:25足を含む未処理の確定足は12:30以降の次サイクルで判定する。
  - ログ: **`[STOCK_V12] STOP_CHECK`**（`confirmed_5m=1` / `stop_profile=`）/ **`STOP_TRIGGER`** / **`STOP executed`**。
- **損切後の再 BUY（P1・現物ユニバース全体）**:
  - `lot_management.evaluate_stopout_reentry` を `_evaluate_new_entry` 前に接続（`can_enter_after_stopout` は bool 互換API）。
  - 価格回復条件は、日足ATRが取得できる場合だけ **`last_stop_price + reentry_distance_atr × 日足ATR`**（既定 0.7）を適用する。上記の公式日足保存では通常ATRが残らないため、現行通常経路ではこの価格条件をスキップする（5分ATRへの代替はしない）。
  - 同日: **`stopout_date`** でも当日再 BUY 禁止（従来どおり）。
  - `max_stopouts_per_day=1`、当日損失予算、NetLiquidation不明時のfail-closed、cooldownはATR欠損時も適用する。
  - 設定: `STOCK_V12_SWING_REENTRY_PARAMS`。
  - STOP 約定時: `on_stopout` → `last_stop_price` / `stopouts_today` 更新。ログ **`stopout_lot_state`** / **`reentry_blocked reason=<個別理由>`**。

**実装の要点**: **`is_open_auction_mask_jst`**（**`SESSION_VWAP_START`〜`OPEN_AUCTION_MASK_END`**）が **BUY（イントラ評価）・`_try_daily_exit_sell`・場中STOP**に効く。旧 **`OPEN_EXIT_*` 定数はコードから削除済み**（寄り窓専用売りに戻さない）。

## GUI・損益表示

- 現物パネルの **「買いシグナル」** は `daily_direction == LONG`、**「売りシグナル」** は `_daily_exit_flag` を表示根拠とする。レガシー5分指標列 `long_condition` / `close_condition` は表示・発注権限を持たない。
- 含み損益は `(現在値 - entry_price) × position` の円表示。`last_trade_info["profit_pct"]` は銘柄ごとの最終SELL 1件であり、日次累計ではない。
- `_realized_loss_today` はSTOP後の再エントリー予算専用で、日足exit損益を含む現物累計台帳ではない。N225 Live Microの週次REAL台帳は現物へ流用しない。

---

## 廃止した旧条件

以下は**もう日足ゲート・イントラでは使わない**。

- `G_OVERHEAT`
- SMA50 / SMA200 **slope** 条件
- **GU ガード**（`C_GAP_UP_LIMIT`）
- **`v12_skip_gap_up_block`**（設定キーも削除）

---

## 実装上の要点

- **daily_direction** は **LONG / NEUTRAL の 2 値**に統一。状態復元・GUI・ログは `_v12_daily_dir_norm` で揃える。
- **stale LONG 防止**: データ不足時は **NEUTRAL**；`_evaluate_new_entry` は **`reg.ok_data` かつ `reg.direction == LONG`**。
- **`process_stock`** / **GUI** / **`save_trading_state` の `daily_direction`** の整合を保つ（保存値も 2 値）。

---

## 関連ファイル（関数・モジュール）

| ファイル | 役割 |
|----------|------|
| `src/trading/stock_vwap_recovery_v12.py` | `evaluate_daily_regime_v12`、`evaluate_intraday_entry_v12`、`evaluate_daily_exit_two_below_sma50`、VWAP・ストップ距離 |
| `src/trading/tradingbot.py` | 注文・reconcile・日足 exit の実行境界 |
| `src/trading/stockbot/v12_daily_gate_mixin.py` | 確定日足の direction / exit gate と公式 `daily_data` 保存 |
| `src/trading/stockbot/strategy_policy_mixin.py` | 新規評価、ストップArm/発火、pending daily exit |
| `src/trading/stockbot/v12_stop_policy_mixin.py` | stop profile、ATR選択、損切後reentry |
| `src/trading/stockbot/trading_loop_mixin.py` | `process_stock` と現物cycle配線 |
| `src/trading/tradingbot_gui.py` | `daily_direction` 表示（既定 `NEUTRAL`） |
| `src/trading/trading_config.py` | `ETF_SETTINGS` / `STOCK_SETTINGS`、`V12_DAILY_GATE_MIN_ROWS` 等 |
| `src/trading/tests/test_stock_vwap_recovery_v12.py` | 純関数テスト |
| `tools/daily_gate_snapshot_v12.py` | ローカル CSV でレジーム確認（`prev_direction=NEUTRAL` 固定の注意あり） |
| `tools/smoke_trading_stage3.py` | 配線スモーク |

---

## レガシー残骸（コード内）

以下は**現行ゲートでは未使用**だが、ファイルに残っている場合がある。

- `stock_vwap_recovery_v12.py`: `GAP_UP_LIMIT`、`OVERHEAT_UPPER_BAND`、`SMA50_SLOPE_WINDOW` / `SMA200_SLOPE_WINDOW` などの**定数**（現行ゲートでは未使用のものあり）。
- **`TradingBot.check_and_execute_loss_cut`**: **建玉×％損切り経路は削除済み**（現物ユニバースは **`stop_line` のみ**）。GUI/別経路用に **`_loss_cut_pct_by_symbol`** 等の属性が残る場合は**参照のみ**。
- 旧ドキュメントに**寄り窓のみ売る**等の文言が残る場合は**誤り**。**本ファイルを正**とする。

---

## 関連ドキュメント

- ユニバース・運用メモ: [`CASH_STOCK_SWING_UNIVERSE.md`](CASH_STOCK_SWING_UNIVERSE.md)
- 場中 ATR ストップ設計: [`STOCK_V12_INTRADAY_STOP_ATR_DESIGN.md`](STOCK_V12_INTRADAY_STOP_ATR_DESIGN.md)
- 索引: [`README.md`](README.md)
- 歴史的長文仕様: [`../../archive/legacy_docs_20260522/stock/`](../../archive/legacy_docs_20260522/stock/) — **本ファイルを優先**
