# Antifragile Thematic Alpha

J-Quants APIを使用した投資銘柄スクリーニングシステム

開発版コード：`1.1.0-rc1-margin-daily`。信用残の日次取得・診断を追加しました。
採点ロジックは `JQ-STD-CORRECTED-1.0-rc1` を維持しています。
[信用残の操作・実装範囲](docs/JQuants_Margin_Daily_Implementation.md) /
[採用計画の原文](doc/archive/JQuants_Margin_Daily_Migration_Plan.md) /
[旧版への復元](archive/jquants-1.0.3-rc1-cli3-all-20260930/README.md)。

## 概要

このプロジェクトは、J-Quants APIから取得した株式データを分析し、投資判断に役立つスクリーニングとランキングを提供します。

## 主な機能

- **データ収集**: レート制限を守り、チェックポイント保存と再開に対応
- **信用残の診断**: 信用残だけの更新と通信なしの診断。価格基準日・採点・新規買い可否を保持
- **凍結キャッシュ**: 取得したデータをローカルに保存し、オフライン分析を可能に
- **投資分析**: Piotroski F-Score、バリュエーション指標、安全性スコアなどを計算
- **スイング向けスクリーニング**: 21〜126営業日保有を前提に、財務品質・安全性・バリュエーションを重視
- **ロジック正本**: `JQ-STD-CORRECTED-1.0-rc1`（J-Quants Standard、追加アドオンなし）

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
python JQuamtsScreeningBot.py --phase analyze --top 20

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
- `docs/JQuants_Standard_Screening_Logic_Master.md`: **現行ロジック正本**（`JQ-STD-CORRECTED-1.0-rc1`）
- `docs/スクリーニング仕様マニュアル.md`: 正本への案内
- `docs/archive/スクリーニング仕様マニュアル_pre-jq-std-1.0-rc1.md`: 旧正本アーカイブ
- `config/theme_tags.yaml`: テーマタグ定義
- `.jquants_cache_v2_standard/`: 既定キャッシュ。信用残は専用 `margin_snapshots/` に保存
- `output_standard/`: 既定出力。信用診断は専用 `margin_history/` に保存

## ライセンス

このプロジェクトは個人利用を目的としています。
















