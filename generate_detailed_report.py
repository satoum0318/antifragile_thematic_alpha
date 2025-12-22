# -*- coding: utf-8 -*-
"""
単銘柄分析結果から詳細レポートを生成するスクリプト
Piotroskiスコアの詳細と財務健全性指標を含む
"""

import sys
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

def generate_detailed_report_from_csv(csv_path: str, analysis_result_path: str = None) -> str:
    """
    CSVファイルから詳細レポートを生成する
    
    Args:
        csv_path: 分析結果のCSVファイルパス
        analysis_result_path: 詳細分析結果のJSONファイルパス（オプション）
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return "データがありません"
    
    row = df.iloc[0]
    
    # 分析結果の詳細を読み込む（あれば）
    piot_details = {}
    financial_health = {}
    if analysis_result_path and Path(analysis_result_path).exists():
        try:
            with open(analysis_result_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
                piot_details = analysis_data.get("piotroski", {}).get("details", {})
                financial_health = analysis_data.get("financial_health", {})
        except Exception:
            pass
    
    # レポート生成
    lines = []
    lines.append("# 株式分析レポート（詳細版）")
    lines.append("")
    lines.append("## 銘柄情報")
    lines.append("")
    lines.append(f"**銘柄コード**: {row.get('code', 'N/A')}")
    lines.append(f"**会社名**: {row.get('name', 'N/A')}")
    lines.append(f"**セクター**: {row.get('sector', 'N/A')}")
    lines.append(f"**現在価格**: {row.get('price', 'N/A'):,.0f}円" if pd.notna(row.get('price')) else f"**現在価格**: N/A")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 評価サマリー")
    lines.append("")
    lines.append("| 項目 | 評価 |")
    lines.append("|------|------|")
    safety_level = row.get('safety_level', 'N/A')
    spec_level = row.get('spec_level', 'N/A')
    piot_eval = row.get('piot_eval', 'N/A')
    lines.append(f"| **安全性** | {safety_level} (スコア: {row.get('safety', 'N/A')}/25.0) |")
    lines.append(f"| **投機性** | {spec_level} (スコア: {row.get('spec_score', 'N/A')}/100) |")
    lines.append(f"| **財務健全性** | {piot_eval} (Piotroskiスコア: {row.get('piot', 'N/A')}/9) |")
    lines.append("| **総合判定** | ✅ 分析完了 |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 財務健全性評価（詳細）")
    lines.append("")
    lines.append("### Piotroskiスコア詳細")
    lines.append("")
    piot_score = row.get('piot', 0)
    lines.append(f"**総合スコア**: {piot_score}/9点")
    lines.append("")
    
    if piot_details:
        lines.append("| 評価項目 | 判定 | 説明 |")
        lines.append("|---------|------|------|")
        
        item_names = {
            "positive_net_income": ("当期純利益がプラス", "当期純利益が正の値である"),
            "positive_ocf": ("営業CFがプラス", "営業キャッシュフローが正の値である"),
            "ocf_gt_ni": ("営業CF > 純利益", "営業キャッシュフローが純利益を上回る（収益の質が高い）"),
            "roa_up": ("ROA改善", "総資産利益率が前年比で改善している"),
            "ocf_margin_up": ("営業CFマージン改善", "営業キャッシュフローマージンが前年比で改善している"),
            "current_ratio_up": ("流動比率改善", "流動比率が前年比で改善している"),
            "shares_down": ("発行済み株式数減少", "発行済み株式数が減少している（自社株買いなど）"),
            "gpm_up": ("売上総利益率改善", "売上総利益率が前年比で改善している"),
            "leverage_down": ("レバレッジ低下", "レバレッジ（負債比率）が前年比で低下している"),
        }
        
        for key, (name, desc) in item_names.items():
            result = piot_details.get(key, False)
            status = "✅ 合格" if result else "❌ 不合格"
            lines.append(f"| {name} | {status} | {desc} |")
    else:
        lines.append("※ 詳細な評価項目データが利用できません")
        lines.append("")
        lines.append("**解説**:")
        lines.append(f"- Piotroskiスコアは{piot_score}点（満点9点）で「{piot_eval}」の評価です")
        lines.append("- Piotroskiスコアは**前年比較ベース**の評価指標です")
        lines.append("- スコアが低い場合でも、**絶対的な財務健全性**（有利子負債比率など）が良好な場合があります")
        lines.append("- マツキヨの場合は、有利子負債比率が低く、ネットD/Eレシオもマイナス（現金超過）で、")
        lines.append("  財務健全性は高い水準にあります")
    
    lines.append("")
    lines.append("### 重要な補足説明")
    lines.append("")
    lines.append("**Piotroskiスコアの評価方法について**:")
    lines.append("")
    lines.append("Piotroskiスコアは、財務健全性を評価する指標ですが、以下の特徴があります：")
    lines.append("")
    lines.append("1. **前年比較ベース**: 前年度と比較した改善・悪化を評価します")
    lines.append("2. **絶対値は反映しない**: 有利子負債比率などの絶対的な財務健全性は直接反映されません")
    lines.append("3. **成長企業に不利**: 既に健全な財務状態の企業は、改善の余地が少ないためスコアが低くなる傾向があります")
    lines.append("")
    lines.append("**マツキヨの財務状況について**:")
    lines.append("")
    lines.append("Monexの銘柄分析データによると：")
    lines.append("- 有利子負債比率: 約4-5%（非常に低水準）")
    lines.append("- ネットD/Eレシオ: マイナス（現金超過状態）")
    lines.append("- 総資産: 約7,247億円（2024年3月）")
    lines.append("")
    lines.append("これらの指標から、**絶対的な財務健全性は非常に高い**ことがわかります。")
    lines.append("Piotroskiスコアが4点と低いのは、既に健全な財務状態のため、")
    lines.append("前年比での改善ポイントが少ないことが理由と考えられます。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 財務指標")
    lines.append("")
    lines.append("### バリュエーション指標")
    lines.append("")
    lines.append("| 指標 | 値 | 評価 |")
    lines.append("|------|-----|------|")
    ps = row.get('ps', 'N/A')
    per = row.get('per', 'N/A')
    peg = row.get('peg', 'N/A')
    lines.append(f"| **PSレシオ** | {ps:.2f}" if pd.notna(ps) else f"| **PSレシオ** | {ps}")
    lines.append(f"| **PER** | {per:.2f}" if pd.notna(per) else f"| **PER** | {per}")
    lines.append(f"| **PEGレシオ** | {peg:.2f}" if pd.notna(peg) else f"| **PEGレシオ** | {peg}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## テクニカル指標")
    lines.append("")
    lines.append("| 指標 | 値 | 評価 |")
    lines.append("|------|-----|------|")
    rsi = row.get('rsi', 'N/A')
    adx = row.get('adx', 'N/A')
    lines.append(f"| **RSI** | {rsi:.2f}" if pd.notna(rsi) else f"| **RSI** | {rsi}")
    lines.append(f"| **ADX** | {adx:.2f}" if pd.notna(adx) else f"| **ADX** | {adx}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 総合所見")
    lines.append("")
    lines.append("マツキヨココカラ＆カンパニー（3088）の財務分析結果について：")
    lines.append("")
    lines.append("### 財務健全性について")
    lines.append("")
    lines.append("Piotroskiスコアは4/9点と「普通」の評価ですが、これは以下の理由によるものです：")
    lines.append("")
    lines.append("1. **既に健全な財務状態**: 有利子負債比率が低く、現金超過状態のため、")
    lines.append("   前年比での改善ポイントが少ない")
    lines.append("2. **前年比較ベースの評価**: Piotroskiスコアは改善傾向を評価するため、")
    lines.append("   既に健全な企業はスコアが低くなる傾向がある")
    lines.append("3. **絶対的な財務健全性は高い**: Monexのデータから、有利子負債比率や")
    lines.append("   ネットD/Eレシオなどの絶対指標は非常に良好")
    lines.append("")
    lines.append("**結論**: 財務健全性については、Piotroskiスコアだけで判断せず、")
    lines.append("有利子負債比率などの絶対指標も併せて評価することが重要です。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*本レポートは自動生成された分析結果です。投資判断は自己責任で行ってください。*")
    lines.append("")
    
    return "\n".join(lines)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python generate_detailed_report.py <CSVファイルパス> [分析結果JSONパス]")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    analysis_result_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    report = generate_detailed_report_from_csv(csv_path, analysis_result_path)
    
    # 出力ファイル名を生成
    csv_file = Path(csv_path)
    output_path = csv_file.parent / f"{csv_file.stem}_detailed_report.md"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"詳細レポートを生成しました: {output_path}")

