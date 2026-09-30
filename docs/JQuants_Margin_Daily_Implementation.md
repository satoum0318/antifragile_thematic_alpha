# 信用残日次対応 第1段階の実装・操作手順

2026-09-30 JST。コード版 `1.1.0-rc1-margin-daily`、ロジック版 `JQ-STD-CORRECTED-1.0-rc1`。
開発ブランチ：`feat/jquants-margin-daily-20260930`。
コード版タグ：`code-v1.1.0-rc1-margin-daily`。

採用した仕様は [JQ-MARGIN-DAILY-PLAN-20260928 の原文](../doc/archive/JQuants_Margin_Daily_Migration_Plan.md)です。ユーザーから受領した添付本文を、指定先へそのまま保存しました。原文の「ドキュメント化のみ」「実データ取得未確認」は2026-09-28時点の記録で、後日の実装指示により第1段階を開発しています。本書が実装状況と操作方法の補足です。

## 実装した範囲

| 対象 | 動作 |
|---|---|
| 通常 `collect` | 日付による取得停止を解除し、同じ `/v2/markets/margin-interest` を検証アダプタへ通す。従来の価格日固定・収集対象選択は保持。 |
| `margin_refresh` | 指定した通常snapshotの普通株を、完了銘柄も含め信用残だけ再取得。通常snapshotを更新せず、独立したsnapshot・再開記録へ保存。 |
| `margin_analyze` | 保存済み信用snapshotと元の価格snapshotだけを読み、通信・認証情報読込・再採点をせず専用診断を出力。 |
| 通常 `analyze_symbol` | 同じ診断定義を使用。通常snapshot内の信用残と通常分析の情報締切だけを使用。専用信用snapshotを暗黙に混ぜない。 |
| メニュー・ランナー | メニュー10で信用残更新、11で信用診断。ランナーも両フェーズに対応。 |

信用残は安全性点・投機性点・旧参考点・推奨点・新規買い可否へ接続していません。`short_selling_change_rate` は従来の空欄を維持します。`update_daily`、価格の日次進行、規制の空売り残高報告、日々公表銘柄の取得仕様は変更対象外です。

責務は `jq_margin.py`（純粋な検証・診断計算）、`jq_margin_workflow.py`（専用収集・履歴・出力）、`JQuamtsScreeningBot.py`（CLIと通常処理への接続）に分かれます。

## 操作

Python 3.10以上、`numpy pandas requests` を使用します。通常snapshotがある既存キャッシュを指定してください。信用残更新は価格や財務データを用意する機能ではありません。

```powershell
# コード版を確認
python -B .\JQuamtsScreeningBot.py --version

# 保存済み通常snapshotの全普通株について信用残だけを更新（通信あり）
python -B .\JQuamtsScreeningBot.py --phase margin_refresh

# 最新の信用残取得試行から診断を出力（通信なし）
python -B .\JQuamtsScreeningBot.py --phase margin_analyze
```

小さい範囲で動作確認する場合は対象コードと元snapshotを固定します。次のIDは自分の保存済みIDへ置き換えます。

```powershell
python -B .\JQuamtsScreeningBot.py --phase margin_refresh --snapshot-id SOURCE_ID --code 8697 --budget 1 --once
python -B .\JQuamtsScreeningBot.py --phase margin_analyze --margin-snapshot-id MARGIN_ID
```

`--cache-dir` と `--output-dir` は通常処理と共通です。既定値は `.jquants_cache_v2_standard` と `output_standard`。`--snapshot-id` は元の通常snapshot、`--margin-snapshot-id` は信用残専用snapshotで、用途を分けています。

`runner.py --phase margin_refresh` / `margin_analyze` でも起動できます。ランナーの通常既定動作はオフライン分析です。存在しなかった `DEFAULT_COLLECT_BUDGET` への参照を修正し、チェックポイントの既定値はBot設定へ委ねています。

## 取得窓、停止、再開

新しい信用収集ジョブの開始日をJSTで D とし、両端込み90暦日 `[D-89, D]` を固定します。例：9月30日開始なら7月3日〜9月30日。価格基準日を取得窓の上端にしません。`code + from + to` で全ページを取得し、`published_date` を併用しません。

通常snapshotのカレンダーで必要期間が不足するときだけ、診断用カレンダーを追加取得します。株価・財務・空売り報告は再取得しません。APIレート制御と収集ロックは通常収集と共有します。

`--budget` は1チェックポイントの銘柄数、`--once` は1チェックポイントで停止、`--max-batches` は回数上限です。指定がなければ未処理キューが空になるまで継続します。銘柄取得失敗は `--max-retries`（既定3）で打ち切り、失敗だけの再試行前は `--retry-delay`（既定60秒）待ちます。

Ctrl+Cでは途中状態も独立snapshotへ保存します。同じコマンドで未処理分を再開します。再開中に日付や通常の最新snapshotが変わっても、元snapshotと取得窓を保持します。取得元・対象銘柄・再試行上限を変更する場合は `--reset-pending` で新しいジョブを始めます。既存の履歴は削除しません。

正常な0行は `no_rows` として取得完了、通信失敗・権限不足・スキーマ不一致は別状態です。未処理・打切り失敗・カレンダー不足があれば更新CLI終了コード2、致命的エラーは1、中断は130です。診断CLIはレポートの作成成功を0で返し、データ不足はレポート内の状態・空欄理由で示します。最新参照先は「最新の試行」であり、今回失敗した銘柄を前回成功値で埋めません。前回観測は別欄で示します。

## 保存形式と確認する時点

```text
<cache>/objects/<hash>.json.gz       原データとアダプタ付加情報（既存形式）
<cache>/margin_snapshots/<id>.json   変更不可の信用専用snapshot
<cache>/latest_margin_snapshot.json 最新の試行への参照
<cache>/margin_collection_resume.json 信用収集専用の再開状態
<output>/margin_history/<id>/
    manifest.json
    margin_diagnostics.csv
    margin_diagnostics.json
    report_margin.md
<output>/latest_margin_run.json     最新の信用診断への参照
```

元snapshot・通常の `latest_snapshot.json`・通常の再開記録・通常の採点出力を変更しません。原レスポンス、未知の追加列、行内容ハッシュ、各版の `first_observed_at` と再観測時刻を保持します。日付が同じでも値が訂正されれば別版です。再取得した過去行に、以前の別内容の取得時刻を流用しません。

診断には元の `source_snapshot_id`、`price_asof_date`、`source_information_cutoff` と、信用診断自身の `information_cutoff` を保存します。申込日 `Date` は営業日であること、日次は `Date < PubDate <= 情報締切の日付`、採用する内容版の初回観測は情報締切以前であることを要求します。

公表時刻が日付しかない行を、公表日の午前0時から知っていたことにはしません。同じ公表日の取得後なら、その保存済み取得証拠により使用できます。週次は公表日不明のまま実際の取得証拠を使います。保存した版より前の過去順位を正確に再現できるとは扱いません。締切を指定する場合は信用snapshotの検証時刻以前に限ります。

価格が9月11日でも、9月25日以降の信用残を現在の診断として表示できます。その場合 `margin_after_price_asof=true` として時点の差を明示します。9月11日時点の順位へ後から流用しません。

## 診断値と空欄

| 主な列 | 定義・制限 |
|---|---|
| `margin_ratio` / `margin_ratio_state` | `LongVol / ShrtVol`。売残0・買残ありは `zero_short_balance`、両方0は `both_zero`、倍率は空欄。 |
| `margin_long` / `margin_short` | 信用買残株数・信用売残株数。報告件数や報告主体の合計で代用しない。 |
| `margin_short_neg` / `margin_short_std` | 一般・制度の信用売残株数。 |
| `margin_general_short_pct` | `100 * ShrtNegVol / ShrtVol`。分母0や一般売残が合計を超える場合は空欄。 |
| `margin_*_value` / `margin_value_ratio` | 金額6列の原値と金額比。単位未確認。円換算や評価価格の推定はしない。金額比は監査用。 |
| `margin_{long,short}_change_{1,5,20}bd_pct` | 対象期間の株数変化率。日次の連続 N+1 観測、同一銘柄・同一対象区分、営業日完全性、取得時点、企業行動確認を要求。 |
| `margin_short_to_volume_20bd` | 申込日までの20営業日の平均出来高に対する信用売残。実際の返済日数・公式回転日数ではない。古い価格日の出来高で代用しない。 |
| `margin_application_age_bd` / `margin_publication_age_days` | 申込日からの営業日経過と、公表日からの暦日経過。カレンダー不明なら営業日経過は空欄。 |
| `margin_bridge_*` | 最終観測週次→初日日次の別指標。始終日と実営業日数を併記し、1営業日変化へ混入させない。 |
| `margin_metric_states` / `margin_metric_reasons` | 各数値の可否と空欄理由。CSVではJSON文字列、JSONでは辞書。 |

欠測を安全や0と扱いません。取得状態、行の検証状態、対象区分、指標ごとの理由を分離します。`IssType=3` は適用対象外、未知区分は未確認です。最新行の問題を古い正常行で隠しません。適用対象外の原値は生データに残し、評価用数値を空欄にします。

株数は遡及調整済みとみなしません。既存価格キャッシュの対象期間について `ExRT`・`AdjFactor` を確認し、分割・併合・ライツ、未知の係数や確認不足があれば変化率は空欄です。価格日が古ければ新しい期間の企業行動は未確認となります。第1段階では追加株価収集や信用残の調整系列を作りません。

9月18日週次→9月25日日次は2営業日分であり、1ステップの1営業日変化ではありません。5営業日変化は最短10月2日申込分（通常10月5日公表）、20営業日は最短10月26日申込分（通常10月27日公表）。それ以前の空欄は正常です。その日以後も価格履歴・企業行動等の条件が欠ければ空欄を維持します。

## 確認済み事項と残る確認

9月30日の先行API確認では、`86970` の期間取得と公表日取得の2要求が成功しました。期間取得には9月18日週次（`PubDate`・金額列null）、9月25日・28日日次が含まれ、`published_date=2026-09-29` は9月28日申込分を返しました。これは仕様の接続確認であり、この開発版による全銘柄の運用監査ではありません。

同日に確認した [Standardの更新タイミング](https://jpx-jquants.com/ja/spec/data-update) は信用残を16:00頃と記載しています。原計画の同ページ未確認状態を更新します。これは配信完了保証ではなく、コードに確定時刻として埋め込みません。[信用残仕様](https://jpx-jquants.com/ja/spec/mkt-margin-int) に基づく日付と取得証拠で判定します。金額の単位・評価価格は未確認のままです。

オフライン検証は以下で実行します。テスト中の通常HTTP通信は禁止し、ページング・失敗・中断は明示的な偽クライアントで再現します。

```powershell
python -B -m pytest -q -p no:cacheprovider tests
```

2026-09-30の実行結果：**82 passed**。旧版との採点・ゲート・順位比較を含みます。主要採点関数、`update_daily`、共通の公表時刻解釈を含む9関数は、アーカイブとのAST比較でも同一でした。旧BotのSHA-256は保存時から不変、計画アーカイブはユーザー添付とバイト一致を確認しています。

検証対象は、週次・日次の境界、欠測と0、訂正の初回観測、日付のみの公表、休日と系列不足、企業行動、古い価格日、取得失敗、ページング、固定窓での中断再開、通常snapshot不変、出力の再現性、旧版との採点・順位比較です。実データの全銘柄確認、訂正到着の長期観測、性能監査は後続の運用確認として残ります。

旧参考点への接続、増減率の閾値決定、採点・新規買い条件の変更、`published_date` による差分取得、企業行動データの追加調達、価格更新、自動スケジューラは後段階です。

## 旧版へ戻す

[アーカイブと復元手順](../archive/jquants-1.0.3-rc1-cli3-all-20260930/README.md)を参照してください。未保存の変更を保存してから、次のブランチに切り替えます。

```powershell
git switch baseline/jquants-v1.0.3-rc1-cli3-all-20260930
python -B .\JQuamtsScreeningBot.py --version
```

復元用タグは `baseline-jquants-v1.0.3-rc1-cli3-all-20260930`。旧Botのファイルも `archive/jquants-1.0.3-rc1-cli3-all-20260930/` に保存しました。コード版 `1.0.3-rc1-cli3-all` を確認してください。Gitは無視対象の認証情報・キャッシュ・出力を復元しません。新しい信用専用履歴は通常snapshotと独立しており、旧版はこれらを読みません。
