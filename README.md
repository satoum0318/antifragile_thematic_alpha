# Antifragile Thematic Alpha

J-Quants APIを使用した投資銘柄スクリーニングシステム

開発版コード：`1.2.0-rc1-universe-ab`。A「割安・財務良好・成長」とB「高成長・信用売残」の登録検討候補を別々に出力します。
登録候補ロジックは `JQ-UNIVERSE-AB-1.0-rc1`。旧 `JQ-STD-CORRECTED-1.0-rc1` の順位も比較用に保存します。
[A・Bの条件・初期値・操作](docs/JQuants_Universe_AB_Implementation.md) /
[直前のrc2版へ復元](archive/jquants-1.1.0-rc2-before-ab-20260930/README.md) /
[信用残の操作・実装範囲](docs/JQuants_Margin_Daily_Implementation.md) /
[採用計画の原文](doc/archive/JQuants_Margin_Daily_Migration_Plan.md) /
[旧版への復元](archive/jquants-1.0.3-rc1-cli3-all-20260930/README.md)。

## 概要

このプロジェクトは、J-Quants APIから取得した株式データを分析し、投資判断に役立つスクリーニングとランキングを提供します。

## 主な機能

- **データ収集**: レート制限を守り、チェックポイント保存と再開に対応
- **信用残の診断**: 買残・売残・変化を時点監査。B候補では有効な売残の積み上がりを確認。メニュー10・11単独では再採点しない
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

### コマンドラインオプション

```bash
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
- `docs/JQuants_Universe_AB_Implementation.md`: **A・B登録候補の現行仕様**（`JQ-UNIVERSE-AB-1.0-rc1`）
- `docs/JQuants_Standard_Screening_Logic_Master.md`: **旧方式比較・基礎データ計算の正本**（`JQ-STD-CORRECTED-1.0-rc1`）
- `docs/スクリーニング仕様マニュアル.md`: 正本への案内
- `docs/archive/スクリーニング仕様マニュアル_pre-jq-std-1.0-rc1.md`: 旧正本アーカイブ
- `config/theme_tags.yaml`: テーマタグ定義
- `.jquants_cache_v2_standard/`: 既定キャッシュ。信用残は専用 `margin_snapshots/` に保存
- `output_standard/`: 既定出力。信用診断は専用 `margin_history/` に保存

## ライセンス

このプロジェクトは個人利用を目的としています。













