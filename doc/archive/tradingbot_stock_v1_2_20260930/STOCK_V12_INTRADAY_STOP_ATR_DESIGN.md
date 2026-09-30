# stock v1.2 場中ストップ ATR ベース再設計（設計整理 + 最小実装案）

**現行実装確認**: 2026-09-30（コードを正として同期）
**release**: app `v3.3.251` / GUI `v1.176` / `stock-vwap-v1.2-code-authority-20260930`

> 本文には段階導入時の設計経緯も残す。現行契約は Step 4b と
> `STOCK_VWAP_V1_2_LOGIC.md` を優先する。

対象: TradingBot 現物ユニバースの **v1.2（`stock_vwap_v1_2`）のみ**。先物・GEX・仮想ETFは対象外。

---

## Step 1: 現行 stop 関係コードの棚卸し

| file | symbol / location | role | 変更要否 |
|------|-------------------|------|----------|
| `stock_vwap_recovery_v12.py` | `SLIP_TICKS` | 固定2ティックの倍数。旧ストップ幅に使用 | **参照片**: 新パラメータ `stop_min_ticks` に吸収可。定数は残しても可 |
| `stock_vwap_recovery_v12.py` | `slip_buffer_from_price` | `jp_price_tick_size * slip_ticks` | **据置可**。テスト・他用途。新ストップ距離の「ティックフロア」計算に再利用可 |
| `stock_vwap_recovery_v12.py` | `jp_price_tick_size` | 呼値単位 | **据置** |
| `stock_vwap_recovery_v12.py` | `entry_bar_stop_line(df5, entry_i, price)` | `low` と `slip_buffer` から旧 `stop_line` | **置換 or 非推奨ラップ**: 新 `compute_intraday_stop_line_atr`（仮名）に統合 |
| `stock_vwap_recovery_v12.py` | `EXEC_STAGE_LATE_END`, `evaluate_intraday_entry_v12` | `close_rush` / `aggressive` の判定元 | **据置**。`is_close_rush` は `execution_stage == "close_rush"` または `fallback_to_aggressive` で派生 |
| `tradingbot.py` | `_evaluate_new_entry` | `entry_bar_stop_line` 呼び出し、`pending_stop` 構築、`process_order(BUY)` | **要変更**: 新 helper 呼び出し、dict 拡張、**STOP_ARM** ログ |
| `tradingbot.py` | `process_order` 成功時 / `_stock_v12_try_promote_stop_on_ib_position` | `pending_stop` → `active_stop` | **要変更（軽微）**: 追加フィールドをそのままコピー |
| `tradingbot.py` | `check_and_execute_loss_cut` | `stop_line` で `close < stop_line` 判定、STOP_CHECK/TRIGGER | **ロジック据置**。ログに `st` の追加キーがあれば出力拡張可 |
| `tradingbot.py` | `save_trading_state` / 起動時 restore | `stock_v12.active_stop` / `pending_stop` JSON | **後方互換**: 新キーは任意。旧キー `stop_line` のみでも発火可能 |


---

## Step 2: 新 stop 仕様（ATR + フロア + close_rush 補正）

### 2.1 入力（ARM 時に揃える）

| 入力 | 説明 |
|------|------|
| `entry_price` | 現行どおり: エントリー判定に使った **5分足 `entry_i` の終値**（`exec5["close"].iloc[entry_i]`） |
| `entry_bar_low` / `entry_bar_high` | `exec5` の同一行の high/low |
| `tick_size` | `jp_price_tick_size(entry_price)`（または実株価に合わせた代表価格） |
| `atr_5m` | 下記「取得方針」 |
| `is_close_rush` | `intr.execution_stage == "close_rush"` **または** `intr.fallback_to_aggressive is True` |

### 2.2 ATR実効値の取り方（現行コード）

1. `swing` は `_stock_v12_daily_atr` の**日足ATR**を最優先する。ただし公式更新 `_set_daily_from_dailydf_stock_v12` は `daily_data` をOHLCV＋SMA50/SMA200へ正規化して保存し、`atr` 列を残さない。このため通常経路では日足ATRは取得できない。
2. 初回BUYは `df["atr"].iloc[entry_i]` の**5分ATR**へフォールバックする。列欠損・NaN・範囲外なら `intr.atr5_used` を使う。
3. 外部/手動建玉のretrofitは最新確定5分足のATRへフォールバックする。
4. すべて取得不能なら **`max(entry_price × params.atr_fallback_pct, 2 × tick_size)`**。現行 `swing` は `atr_fallback_pct=0.015`、レガシー `intraday` は `0.003`。

変数名・保存キーは後方互換のため `atr_5m` / `atr_5m_arm` のままだが、値の出所は `atr_source` を正とする。

### 2.3 `stop_line` 計算（純関数・v12 モジュール）

**Anchor**

- `anchor = min(entry_price, entry_bar_low)`

**`entry_bar_range`**

- `entry_bar_high - entry_bar_low`（負・0 は 0 扱いでよい）

**`stop_distance`（ベース）**

\[
\text{stop\_distance} = \max\left(
\begin{array}{l}
\text{stop\_min\_ticks} \cdot tick\_size \\
\text{stop\_min\_pct} \cdot entry\_price \\
\text{stop\_bar\_range\_mult} \cdot entry\_bar\_range \\
\text{stop\_atr\_mult} \cdot atr\_{eff}
\end{array}
\right)
\]

- `atr_eff`: 上記 2.2 で決めた有限正の値。

**`close_rush` 補正**

- `stop_distance_adj = stop_distance * close_rush_stop_mult`
- さらにフロア: `stop_distance_final = max(stop_distance_adj, close_rush_min_pct * entry_price)`（close_rush 時**のみ**後段 max を適用してよい）
- **非 close_rush** では `stop_distance_final = stop_distance`（mult / min_pct は適用しない）

**`stop_line`**

- `stop_line = anchor - stop_distance_final`
- オプション（実装フェーズ2）: `stop_line` を呼値に合わせ `adjust_to_tick_size` で **下方向** に丸め、過剰にきつくならないよう注意。

### 2.4 発火条件（据置）

- `latest_5m_close < stop_line`（現行 `check_and_execute_loss_cut` のまま）
- 形成中足は使わず、09:00–09:05 JST はSTOP発注を保留して `active_stop` を維持する。
- 昼休みは `is_market_open=False` のため、11:25足を含む未処理の確定足は12:30以降の次サイクルで判定する。

### 2.5 `pending_stop` / `active_stop` に保存するフィールド

**必須（現行互換・発火に必要）**

- `stop_line` float
- `entry_bar_low` float（ログ・デバッグ）
- `entry_i` int
- `entry_price_ref` float

**推奨追加（デバッグ・事後分析・将来の再現）**

- `anchor` float
- `stop_distance` float（`stop_distance_final`）
- `atr_5m_arm` float（実際に使った ATR）
- `entry_bar_high` float
- `is_close_rush` bool
- `stop_version` str 例 `"atr_v1"`（旧 state に無い場合は読み手が「レガシー」と判断可能）

### 2.6 後方互換

- **永続化 JSON**: 旧データは `stop_line` + 既存キーのみ。→ **ロード後もそのまま発火可能**。新キーは無くてよい。
- **コード**: 古い `stop_line` だけの position を再計算しない（**約定済みラインを尊重**）。
- **テスト**: 旧形式 dict を `active_stop` に載せたとき `STOP_CHECK` が動くこと。

---

## Step 3: 最小実装案

### 3.1 `stock_vwap_recovery_v12.py`

1. **`@dataclass StockV12StopParams`**: 下記 Step 4 の既定値を保持（モジュール定数から1インスタンスでも可）。
2. **`@dataclass StockV12StopArmResult`**（frozen）: `stop_line`, `anchor`, `stop_distance_final`, `atr_eff_used`, `entry_bar_range`, `is_close_rush`, components dict（各 max の内訳はログ用 tuple で十分）。
3. **`compute_intraday_stop_arm_atr(...)`** → `StockV12StopArmResult`  
   - 引数: `entry_price`, `entry_bar_low`, `entry_bar_high`, `atr_5m`, `is_close_rush`, `params`, `tick_size`（省略時は `jp_price_tick_size(entry_price)`）。
4. **`entry_bar_stop_line`**:  
   - 案A: 削除せず **deprecated コメント** + 内部で新関数を呼ぶためのブリッジ（テスト互換）。  
   - 案B: **残置のみ**（呼ばれなくなる）。推奨は **案B + 新関数のみ tradingbot から呼ぶ**（責務が明確）。

### 3.2 `tradingbot.py`

1. `_evaluate_new_entry` で `entry_i` 確定後:
   - `eh = exec5["high"].iloc[entry_i]`, `el = exec5["low"].iloc[entry_i]`
   - `atr_5m` を 2.2 で取得
   - `is_close_rush = intr.execution_stage == "close_rush" or intr.fallback_to_aggressive`
   - `arm = stock_vwap_v12.compute_intraday_stop_arm_atr(...)`
   - `pending_stop` に `arm` と既存キーをマージ
   - **`[STOCK_V12] STOP_ARM ...`** 1 行ログ
2. `check_and_execute_loss_cut`: **条件式は変更なし**。STOP_CHECK に `anchor` / `stop_distance` 等があれば付加ログ（任意）。

### 3.3 テスト

- `test_stock_vwap_recovery_v12.py`:  
  - 1417 類似: entry 4030, low 4031, high 4032, atr 小さめ → 旧より `stop_line` が **下がる**（距離が広がる）ケース  
  - close_rush で `close_rush_min_pct` が効くケース

---

**Phase 2 実装済み（2026-06-26）**: `trading_config.STOCK_V12_STOP_PROFILE_BY_PRODUCT` / `STOCK_V12_STOP_PARAMS_BY_PROFILE`。現物ユニバース全体=`swing`（個別株・ETF 共通）。

---

## Step 4b: スイングプロファイル（現物ユニバース全体）

**スコープ**: TradingBot 現物の **個別株・現物ETF**。FuturesBot / 先物デイトレとは無関係。

| 項目 | intraday（レガシー参照） | swing（現物ユニバース既定） |
|------|----------------|----------------|
| `stop_version` | `atr_v1` | `swing_v1` |
| ATR 入力 | 5 分 ATR | 日足ATRを優先するが、現行公式保存では通常欠落するため **5分ATRへフォールバック** |
| `stop_atr_mult` | 0.35 | 1.0 |
| `stop_min_pct` | 0.003 | 0.015 |
| `stop_bar_range_mult` | 0.5 | 0.25 |
| 発火 | 確定 5 分 close | 同左 |

**再エントリー（P1）**: `STOCK_V12_SWING_REENTRY_PARAMS` + `lot_management.evaluate_stopout_reentry`（`can_enter_after_stopout` はbool互換API）。日足ATRがある場合だけ `last_stop_price + 0.7×日足ATR` の価格回復条件を適用する。現行公式保存では通常ATRが欠落するため価格条件はスキップし、同日禁止・1日1回・損失予算等を適用する。再エントリー側は5分ATRへ代替しない。

---

## Step 4: パラメータ一覧・外出し方針

| 名前 | 例（ユーザー案） | 意味 |
|------|------------------|------|
| `stop_min_ticks` | 2 | 最小ティック数フロア |
| `stop_min_pct` | 0.0030 | 株価に対する最小割合フロア |
| `stop_bar_range_mult` | 0.5 | エントリー足レンジの係数 |
| `stop_atr_mult` | 0.35 | ATR 倍率 |
| `close_rush_stop_mult` | 1.5 | close_rush 時の距離倍率 |
| `close_rush_min_pct` | 0.0045 | close_rush 時の割合フロア（max の第2段） |

**今回のコード配置（推奨段階）**

1. **Phase 1（最小）**: `stock_vwap_recovery_v12.py` 内の **モジュールレベル dataclass / 定数**（デフォルト1セット）。`tradingbot` は `stock_vwap_v12.DEFAULT_STOP_PARAMS` のみ参照。
2. **Phase 2**: `trading_config.py` に `STOCK_V12_STOP_DEFAULTS: dict` または `StockV12StopParams` 相当の読み出し（`config.ini` は任意）。
3. **Phase 3**: 銘柄別 `STOCK_V12_STOP_OVERRIDES: dict[str, PartialDict]`（未設定銘柄はデフォルト）。

**現行状態**: Phase 2まで実装済み。商品別profileは stock / ETF とも `swing`、銘柄別overrideは未導入。

---

## Step 5: ログ案

既存を維持しつつ追加。

| ログキー | レベル | 内容 |
|----------|--------|------|
| `[STOCK_V12] STOP_ARM` | INFO | `symbol=... anchor=... entry_price_ref=... entry_low=... entry_high=... atr_5m=... tick_size=... stop_distance=... stop_line=... is_close_rush=... stage=...`（`stop_distance` は final） |
| `[STOCK_V12] STOP_CHECK` | INFO | 現行維持 + 可能なら `anchor= stop_distance=` を `active_stop` から |
| `[STOCK_V12] STOP_TRIGGER` | WARNING | 現行維持 |

---

## Step 6: やらないこと（再掲）

- broker 逆指値の全面載せ替え  
- 発火をティック/1分足へ変更  
- 日足ゲート・BUY 条件・VWAP 本体・close_rush エントリー本体の再設計  

---

## Step 7: backward compatibility 注意点

- 状態ファイルに **新キーが無い** 旧 `active_stop` はそのまま有効。
- 新コードが **旧キーのみ** を読んだとき、STOP_CHECK の拡張フィールドは出さない。
- `stop_version` は任意。将来「再ARM」ロジックを入れるなら有効。

---

## Step 8: `git diff --stat` 見込み（目安）

```
 src/trading/stock_vwap_recovery_v12.py | 110-150 +++++
 src/trading/tradingbot.py              |  55-80 +-
 src/trading/test_stock_vwap_recovery_v12.py | 50-80 ++++
 docs/STOCK_V12_INTRADAY_STOP_ATR_DESIGN.md   | 本ファイル
```

（`trading_config.py` を Phase 1 で触らなければ 0。Phase 2 で +20 行前後。）

---

## Step 9: 実装順（小さく分ける）

1. **PR/A**: `stock_vwap_recovery_v12.py` — `StockV12StopParams`, `StockV12StopArmResult`, `compute_intraday_stop_arm_atr` + ユニットテスト（1417 風ケース・close_rush）。
2. **PR/B**: `tradingbot._evaluate_new_entry` — 新 helper 利用、`pending_stop` 拡張、`STOP_ARM` ログ。
3. **PR/C**（任意）: `check_and_execute_loss_cut` の STOP_CHECK  enriched ログ。
4. **PR/D**（後日）: `trading_config` からの読み込み・銘柄別 override。

---

## 参考: 1417 事例でのイメージ（数値イメージのみ）

- `entry_price≈4030`, `entry_bar_low=4031`, `anchor=4030`  
- 旧: `stop_line=4029`（約1円）  
- 新: 例 `atr_eff≈6`, `stop_atr_mult=0.35` → 2.1 円、他フロアと max により **少なくとも数円〜10円台前半** 程度の距離になりやすく、即死しにくい（実数はパラメータ次第）。

---

*最終更新: 設計整理段階（未実装）。実装時は本書 Step 3/9 に従い小分け PR を推奨。*
