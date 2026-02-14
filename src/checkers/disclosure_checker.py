"""
重要事項説明書の入力ミスチェック
- 必須項目の有無
- 数値・日付のフォーマット
"""
import re
from .base import BaseChecker, CheckResult, Severity


class DisclosureChecker(BaseChecker):
    """重要事項説明書の入力チェック"""

    @property
    def name(self) -> str:
        return "重要事項説明書チェック"

    # 重要事項説明でよく出るキーワード（あれば必須項目の参照用）
    KEYWORDS = [
        "重要事項の説明",
        "取引態様",
        "登記",
        "権利の種類",
        "法令上の制限",
        "私道負担",
        "設備",
        "支払金",
        "契約解除",
        "損害賠償",
    ]

    def run(self, text: str) -> list[CheckResult]:
        results: list[CheckResult] = []
        # 重要事項説明書らしいキーワードの有無（参考）
        found = [k for k in self.KEYWORDS if k in text]
        if len(found) < 3:
            results.append(CheckResult(
                severity=Severity.INFO,
                category="書類種別",
                message="重要事項説明書として認識される項目が少ないです",
                detail="契約書や設備表の可能性があります。書類を確認してください。",
            ))
        # 取引態様の表記
        if "取引態様" in text:
            for m in re.finditer(r"取引態様[：:]\s*([^\n]+)", text):
                val = m.group(1).strip()
                if not val or len(val) < 2:
                    results.append(CheckResult(
                        severity=Severity.WARNING,
                        category="取引態様",
                        message="取引態様の記載が空または短いです",
                        detail=val or "(空)",
                        position=_nearby(text, m.start(), 50),
                    ))
        # 金額・日付の不正（契約書と同様の簡易チェック）
        for m in re.finditer(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日", text):
            month, day = int(m.group(1)), int(m.group(2))
            if month < 1 or month > 12:
                results.append(CheckResult(
                    severity=Severity.ERROR,
                    category="日付",
                    message="月が不正です",
                    detail=m.group(0),
                ))
            if day < 1 or day > 31:
                results.append(CheckResult(
                    severity=Severity.ERROR,
                    category="日付",
                    message="日が不正です",
                    detail=m.group(0),
                ))
        # 空欄・未記入
        if "（　）" in text or "（  ）" in text:
            results.append(CheckResult(
                severity=Severity.WARNING,
                category="空欄",
                message="空欄「（　）」が含まれています。必要項目の記入を確認してください。",
            ))
        return results


def _nearby(text: str, pos: int, length: int) -> str:
    start = max(0, pos - length)
    end = min(len(text), pos + length)
    return text[start:end].replace("\n", " ").strip()
