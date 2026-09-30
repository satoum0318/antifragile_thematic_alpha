# Antifragile Thematic Alpha

J-Quants APIを使用した投資銘柄スクリーニングシステム

開発版コード：`1.3.0-rc1-guided-screening`。通常メニューの「1」でデータ更新からA・B候補の出力まで実行します。
登録候補ロジックは `JQ-UNIVERSE-AB-1.0-rc1`。旧 `JQ-STD-CORRECTED-1.0-rc1` の順位も比較用に保存します。
[A・Bの条件・初期値・操作](docs/JQuants_Universe_AB_Implementation.md) /
[A・B導入前のrc2版へ復元](archive/jquants-1.1.0-rc2-before-ab-20260930/README.md) /
[信用残の操作・実装範囲](docs/JQuants_Margin_Daily_Implementation.md) /
[採用計画の原文](doc/archive/JQuants_Margin_Daily_Migration_Plan.md) /
[旧版への復元](archive/jquants-1.0.3-rc1-cli3-all-20260930/README.md)。

## 概要

このプロジェクトは、J-Quants APIから取得した株式データを分析し、投資判断に役立つスクリーニングとランキングを提供します。

## 主な機能

- **データ収集**: レート制限を守り、チェックポイント保存と再開に対応
- **一括実行**: 初回収集、新規銘柄の補充、価格・財務の日付更新、30日を目安とした履歴再取得、信用残更新、時点確認、採点を順に実行
- **信用残の診断**: 買残・売残・変化を時点監査。B候補では有効な売残の積み上がりを確認
- **凍結キャッシュ**: 取得したデータをローカルに保存し、オフライン分析を可能に
- **投資分析**: Standard財務品質7項目（原著Piotroski F-Scoreとは別）、バリュエーション指標、安全性参考点などを計算
- **登録候補スクリーニング**: 成長根拠を確認し、上昇継続・条件待ちを含めてA・B別に比較。資金配分と銘柄登録は本人が管理
- **旧方式比較**: 従来のFE・新規候補条件・旧Topを保持。注文・決済・外部売買Botの変更なし

## セットアップ

### 必要な環境

- Python 3.10以上
- J-Quants APIアカウント

### インストール

```bash
pip install pandas numpy requests
```

### 設定

`api.ini`ファイルを作成し、J-Quants APIの認証情報を設定してください：

```ini
[DEFAULT]
API_KEY=your_v2_api_key
```

環境変数 `JQUANTS_API_KEY` または `JQ_API_KEY` でも指定できます。オフライン分析では認証情報を読みません。

## 使用方法

### 基本的な実行

```bash
# インタラクティブモード
python JQuamtsScreeningBot.py

# または、ランナーを使用
python runner.py
```

普段は **起動 →「1：最新データに更新してスクリーニング」→ 完了を待つ → A・B上位候補を確認** だけです。
毎日でも数日空いても同じ操作です。途中で中断した場合も同じ「1」で再開します。初回や履歴の定期再取得では時間がかかります。処理段階と30秒ごとのハートビートで稼働を確認できます。

| 通常メニュー | 用途 |
|---|---|
| 1 | 最新データの取得からA・Bランキング作成まで一括実行（API通信あり） |
| 2 | 保存済みの前回ランキングを表示（通信なし） |
| 3 | 保存済みの単銘柄結果を表示（通信なし） |
| 4 | 更新状況・未評価の理由を確認（通信なし） |
| 5 | 詳細操作。旧メニュー1〜12はここへ移動 |

「財務確認日時」と「財務全履歴の最終取得日時」は別です。新しい決算・予想・決算予定は実行ごとの日付更新で取得し、過去の訂正は全履歴の定期再取得で確認します。30日はこのBotの運用設定であり、JPXの公表頻度ではありません。

実行時点で提供されているデータを使います。前営業日分を確認する翌朝が目安です。取得完了は利益や候補10件を保証しません。更新失敗時は前回の結果を残し、今回の最新結果として表示しません。個別の不足がある部分結果は件数と理由を明示します。

### 結果の表示とMarkdown保存

通常メニュー「1」で作成したA・Bレポートは、**Markdownファイルへ保存し、同じ内容をターミナルにも表示**します。保存のために別のメニューを選ぶ必要はありません。既定の保存先は次のとおりです。

```text
output_standard/
  latest_run.json                 最新の公開済み結果の参照先
  history/
    <実行ID>/
      report_registration_top10.md A・B候補、採用根拠、懸念点、信用需給
```

`--output-dir` を指定している場合は、その出力先の `history/<実行ID>/` に保存します。実行ごとに別のフォルダーを作るため、過去のレポートも残ります。Markdownファイルはエディターで開けます。VS Codeでは開いた後に `Ctrl+Shift+V` でプレビューできます。

完了後は「2」で保存済みレポートを再表示し、詳しく見たい銘柄は「3」で銘柄コードを入力します。選ぶ銘柄はご自身で判断して別の売買Botへ登録し、確認が終わったら `q` で終了してください。次回、最新情報で選び直すときは再び「1」です。

候補が0件または10件未満でも、追加の分析操作で件数を埋める必要はありません。Bは有効な20営業日の売残変化を優先し、未計算なら5営業日の変化を使います。日次信用残の開始直後はどちらも計算できず、履歴不足になります。同日の「1」の繰り返しでは履歴の日数は増えません。レポートの条件未達・未評価理由を確認してください。

現在のコード実装タグは `code-v1.3.0-rc1-guided-screening`、運用確認とこの案内を含む文書更新タグは `release-v1.3.0-rc1-guided-screening` です。[リリースノート](CHANGELOG.md) に保存結果の確認記録を記載しています。

### コマンドラインオプション

```bash
# メニューを使わず、一括更新・スクリーニングを実行（API通信あり）
python JQuamtsScreeningBot.py --phase screen

# 前回の保存済みレポートを表示（通信なし）
python JQuamtsScreeningBot.py --phase show-report

# データ収集
python JQuamtsScreeningBot.py --phase collect --budget 800

# 分析実行
python JQuamtsScreeningBot.py --phase analyze --top 10

# 保存済み信用残の最新試行を明示してA・B分析（通信なし、価格・財務の締切は保持）
python JQuamtsScreeningBot.py --phase analyze --margin-snapshot-id latest --top 10

# 保存済み通常snapshotを元に、信用残だけを更新（実API通信あり）
python JQuamtsScreeningBot.py --phase margin_refresh

# 最新の信用残取得試行から診断を出力（通信なし）
python JQuamtsScreeningBot.py --phase margin_analyze

# fins/summary の生フィールド監査
python JQuamtsScreeningBot.py --phase fields-audit --budget 50
```

## ファイル構成

- `runner.py`: 便利なランナースクリプト
- `JQuamtsScreeningBot.py`: J-Quants V2 Standard オフライン一括スクリーニング
- `jq_universe.py` / `jq_universe_report.py`: A・Bの登録候補判定と根拠レポート
- `jq_screening_workflow.py` / `jq_screening_ui.py`: 一括更新・再開・定期再取得と通常メニュー
- `docs/JQuants_Universe_AB_Implementation.md`: **A・B登録候補の現行仕様**（`JQ-UNIVERSE-AB-1.0-rc1`）
- `docs/JQuants_Standard_Screening_Logic_Master.md`: **旧方式比較・基礎データ計算の正本**（`JQ-STD-CORRECTED-1.0-rc1`）
- `docs/スクリーニング仕様マニュアル.md`: 正本への案内
- `docs/archive/スクリーニング仕様マニュアル_pre-jq-std-1.0-rc1.md`: 旧正本アーカイブ
- `config/theme_tags.yaml`: テーマタグ定義
- `.jquants_cache_v2_standard/`: 既定キャッシュ。信用残は専用 `margin_snapshots/` に保存
- `output_standard/`: 既定出力。信用診断は専用 `margin_history/` に保存

## ライセンス

このプロジェクトは個人利用を目的としています。











