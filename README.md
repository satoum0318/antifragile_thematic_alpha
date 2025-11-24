# Antifragile Thematic Alpha

J-Quants APIを使用した投資銘柄スクリーニングシステム

## 概要

このプロジェクトは、J-Quants APIから取得した株式データを分析し、投資判断に役立つスクリーニングとランキングを提供します。

## 主な機能

- **データ収集**: J-Quants APIから1日380銘柄ずつ価格データと財務データを取得
- **凍結キャッシュ**: 取得したデータをローカルに保存し、オフライン分析を可能に
- **投資分析**: Piotroski F-Score、バリュエーション指標、安全性スコアなどを計算
- **テーマ別スクリーニング**: aging_dx_alphaプロファイルによる高齢化×DXテーマ銘柄の選別

## セットアップ

### 必要な環境

- Python 3.7以上
- J-Quants APIアカウント

### インストール

```bash
pip install pandas numpy requests pyyaml
```

### 設定

`api.ini`ファイルを作成し、J-Quants APIの認証情報を設定してください：

```ini
[DEFAULT]
MAIL_ADDRESS=your_email@example.com
PASSWORD=your_password
```

## 使用方法

### 基本的な実行

```bash
# インタラクティブモード
python real_data_screening.py

# または、ランナーを使用
python runner.py
```

### コマンドラインオプション

```bash
# データ収集（1日380銘柄）
python real_data_screening.py --phase collect --budget 380

# 分析実行
python real_data_screening.py --phase analyze --top 20

# aging_dx_alphaプロファイルで分析
python runner.py --phase analyze --profile aging_dx_alpha --top 20
```

## ファイル構成

- `real_data_screening.py`: メインスクリーニングスクリプト
- `runner.py`: 便利なランナースクリプト
- `JQuamtsScreeningBot.py`: 旧バージョン（参考用）
- `config/theme_tags.yaml`: テーマタグ定義
- `.jquants_cache/`: データキャッシュディレクトリ
- `output/`: 分析結果出力ディレクトリ

## ライセンス

このプロジェクトは個人利用を目的としています。

