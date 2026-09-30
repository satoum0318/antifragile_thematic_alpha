# 日次信用残対応前のBotアーカイブ

作成日：2026-09-30（JST）

このディレクトリの `JQuamtsScreeningBot.py` は、日次信用残対応の開発を始める前の現行スクリプトを、バイト列を変更せずコピーしたものです。ファイル名はリポジトリの実名 `JQuamtsScreeningBot.py` を維持しています。

## 保全した版

| 項目 | 値 |
|---|---|
| `CODE_VERSION` | `1.0.3-rc1-cli3-all` |
| `LOGIC_VERSION` | `JQ-STD-CORRECTED-1.0-rc1` |
| `CACHE_SCHEMA` | `jq-standard-raw-v1` |
| 元ブランチ | `feat/ps-only-satellite` |
| 保全元コミット | `ab3a0255a47aaf60f764518fbcad7d64854ca58a` |
| 復元用ブランチ | `baseline/jquants-v1.0.3-rc1-cli3-all-20260930` |
| 復元用タグ | `baseline-jquants-v1.0.3-rc1-cli3-all-20260930` |
| 新規開発ブランチ | `feat/jquants-margin-daily-20260930` |
| スクリプトSHA-256 | `52B9731E6801B194C265F5A657143D0F62C2B9523108C0804EA2A93F32F42228` |

復元用ブランチ・タグは保全元コミットを指します。アーカイブ作成時に、元ファイルとコピーのSHA-256が一致することを確認しました。元ブランチの履歴は変更していません。

`CHANGELOG.md` のリリース番号とスクリプト内のコード版は別の識別子です。復元対象は上記コミットとコード版で特定してください。

## リポジトリ全体を旧版へ戻す

リポジトリ直下で、作業中の変更を確認し、必要な変更をコミットまたは別途保存してから切り替えます。

```powershell
git status --short
git switch baseline/jquants-v1.0.3-rc1-cli3-all-20260930
python -B .\JQuamtsScreeningBot.py --version
```

コード版 `1.0.3-rc1-cli3-all` とロジック版 `JQ-STD-CORRECTED-1.0-rc1` が表示されることを確認します。開発を再開する場合は次を使います。

```powershell
git switch feat/jquants-margin-daily-20260930
```

ローカル変更が原因で切替を拒否された場合は、その変更を保存してから再実行します。復元のために `--force` や `reset --hard` を使う必要はありません。保全コミットから別の作業ブランチを作る場合は、復元用タグを起点にできます。

## 開発ブランチのまま旧スクリプトを使う

以下はリポジトリ直下から実行します。アーカイブ版は `__file__` の位置が変わるため、認証iniは明示的に指定します。

```powershell
python -B .\archive\jquants-1.0.3-rc1-cli3-all-20260930\JQuamtsScreeningBot.py --version
python -B .\archive\jquants-1.0.3-rc1-cli3-all-20260930\JQuamtsScreeningBot.py --api-ini .\api.ini --phase analyze --cache-dir .\.jquants_cache_v2_standard --output-dir .\output_standard_restored
```

`analyze` は保存済みデータを使うオフライン処理です。上記の分析コマンドは手順の記載であり、アーカイブ作成時には実行していません。独自設定を使っていた場合は、元と同じ設定ファイルを `--config` で指定してください。ランナーを含む周辺コードも完全に戻す場合は、ファイルコピーではなく復元用ブランチを使ってください。

Gitは無視対象のAPIキー、設定、キャッシュ、分析結果を復元しません。アーカイブ作成時にこれらは変更していません。今後の開発でも旧版の復元に必要なデータを不可逆に変更せず、新しい保存形式・出力は旧版用と分けてください。

## 保全時の確認結果

- 元スクリプトとアーカイブコピーのSHA-256一致を確認。
- アーカイブ版の `--version` が正常終了し、上記コード版・ロジック版を表示。
- 変更前の既存テストを `python -B -m pytest -q -p no:cacheprovider tests` で実行し、**9件失敗**。
- 失敗原因は、現行コードに存在しない旧関数への参照による `AttributeError`。対象は `fetch_prices_v2_with_meta`、`compute_shareholder_return_quality`、`build_split_adjustment_factors`、`resolve_forward_guidance`、`compute_recommendation_score`。
- 上記失敗は日次信用残対応の変更前に発生した既存の不整合。保全版を「全テスト合格済み」とは扱わない。
- アーカイブ作成ではデータAPIの呼び出し・既存テストの書換え・スクリーニング本体の変更を行っていない。

## 実装仕様の確認状況

アーカイブ作成時は指定計画の所在未確認でした。その後、ユーザーの添付原文と指定パスを確認し、[採用計画](../../doc/archive/JQuants_Margin_Daily_Migration_Plan.md)を保存しました。第1段階の実装は開発ブランチでコード版 `1.1.0-rc1-margin-daily` として進めています。[実装・操作手順](../../docs/JQuants_Margin_Daily_Implementation.md)に開発版の検証範囲を記載します。このディレクトリのBot本体は変更していません。
