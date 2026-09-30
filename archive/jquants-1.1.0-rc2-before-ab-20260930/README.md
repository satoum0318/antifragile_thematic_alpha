# A・Bランキング導入前の復元用アーカイブ

2026-09-30 JST。コード版 `1.1.0-rc2-margin-daily`、ロジック版 `JQ-STD-CORRECTED-1.0-rc1`。
保全元コミットは `2c8a00f`、復元用タグ・ブランチは以下です。

- タグ：`baseline-jquants-v1.1.0-rc2-before-ab-20260930`
- ブランチ：`baseline/jquants-v1.1.0-rc2-before-ab-20260930`
- 次版の開発ブランチ：`feat/jquants-value-growth-swing-20260930`

Bot本体の実際の名前は `JQuamtsScreeningBot.py` です。依存する `jq_margin.py`、`jq_margin_workflow.py`、`runner.py` も同じ版で保存しています。4ファイルは元とバイト単位で一致し、SHA-256は [SHA256SUMS.txt](SHA256SUMS.txt) に記録しました。認証情報・キャッシュ・分析結果はコピーしていません。

## Gitで戻す

稼働中のBotはCtrl+Cで終了・保存を待ち、未コミットの変更があれば保存してから実行します。

```powershell
Set-Location C:\project\antifragile_thematic_alpha
git status --short
git switch baseline/jquants-v1.1.0-rc2-before-ab-20260930
python -B .\JQuamtsScreeningBot.py --version
```

コード版 `1.1.0-rc2-margin-daily` を確認します。Gitは無視対象のキャッシュやAPIキーを復元しません。

## コピペで戻す

新しいコードに未保存の編集があれば先に別途保存してください。以下はリポジトリ直下から実行し、4ファイルを同じ版へ戻します。

```powershell
Set-Location C:\project\antifragile_thematic_alpha
$restoreSource = Join-Path (Get-Location) 'archive\jquants-1.1.0-rc2-before-ab-20260930'
foreach ($restoreName in @('JQuamtsScreeningBot.py', 'jq_margin.py', 'jq_margin_workflow.py', 'runner.py')) {
    Copy-Item -LiteralPath (Join-Path $restoreSource $restoreName) -Destination (Join-Path (Get-Location) $restoreName) -Force
}
python -B .\JQuamtsScreeningBot.py --version
```

新仕様専用の追加モジュールが残っていても、この旧版からは呼びません。独自設定を使う場合は旧版で利用していた設定に戻してください。設定項目を含めリポジトリ全体を復元する場合はGitの復元用ブランチを使用します。

## 上書きせずに旧版を起動する

```powershell
python -B .\archive\jquants-1.1.0-rc2-before-ab-20260930\JQuamtsScreeningBot.py --version
python -B .\archive\jquants-1.1.0-rc2-before-ab-20260930\JQuamtsScreeningBot.py --api-ini .\api.ini --cache-dir .\.jquants_cache_v2_standard --output-dir .\output_standard_restored
```

通常の起動と同じくメニューが表示されます。同じ保存先を使用するBotを同時に起動せず、収集を終えてから切り替えてください。アーカイブのコードは以降変更しません。
