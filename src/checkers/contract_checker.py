"""
契約書の入力ミスチェック
- 金額の桁・表記ゆれ
- 日付の整合性・フォーマット
- 空欄・明らかな誤字の検出
"""
import re
from .base import BaseChecker, CheckResult, Severity


class ContractChecker(BaseChecker):
    """売買契約書などの入力チェック"""

    @property
    def name(self) -> str:
        return "契約書チェック"

    def run(self, text: str) -> list[CheckResult]:
        results: list[CheckResult] = []
        # 金額パターン（円・万円・千円）
        amount_yen = re.findall(r"([0-9,，]+)\s*円", text)
        amount_man = re.findall(r"([0-9,，]+)\s*万円", text)
        # 桁抜けの疑い: 1〜2桁の数字＋万円は要確認
        for m in re.finditer(r"(\d{1,2})\s*万円", text):
            val = int(m.group(1))
            if val < 10 and val > 0:
                results.append(CheckResult(
                    severity=Severity.WARNING,
                    category="金額",
                    message="万円の桁が少ない可能性があります（桁抜けの確認を推奨）",
                    detail=f"「{m.group(0)}」",
                    position=_nearby(text, m.start(), 40),
                ))
        # 日付パターン（和暦・西暦）
        date_patterns = [
            r"令和\s*(\d{1,2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日",
            r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日",
        ]
        for pat in date_patterns:
            for m in re.finditer(pat, text):
                g = m.groups()
                if len(g) == 3:
                    month, day = int(g[-2]), int(g[-1])
                    if month < 1 or month > 12:
                        results.append(CheckResult(
                            severity=Severity.ERROR,
                            category="日付",
                            message="月が不正です（1〜12の範囲）",
                            detail=m.group(0),
                            position=_nearby(text, m.start(), 30),
                        ))
                    if day < 1 or day > 31:
                        results.append(CheckResult(
                            severity=Severity.ERROR,
                            category="日付",
                            message="日が不正です（1〜31の範囲）",
                            detail=m.group(0),
                            position=_nearby(text, m.start(), 30),
                        ))
        # 明らかな空欄・プレースホルダ
        placeholders = ["（　）", "（  ）", "＿＿＿", "___", "未記入", "未定"]
        for ph in placeholders:
            if ph in text:
                results.append(CheckResult(
                    severity=Severity.WARNING,
                    category="空欄・未記入",
                    message=f"未記入・プレースホルダの可能性: 「{ph}」",
                    detail="契約前に記入漏れがないか確認してください。",
                ))
        # 金額のカンマ桁数（3桁区切りでない表記）
        for m in re.finditer(r"([0-9]{4,})\s*円", text):
            s = m.group(1).replace(",", "").replace("，", "")
            if len(s) >= 4 and "," not in m.group(1) and "，" not in m.group(1):
                results.append(CheckResult(
                    severity=Severity.INFO,
                    category="金額",
                    message="円の表記にカンマがありません（読みやすさの確認）",
                    detail=m.group(0),
                ))
        return results


def _nearby(text: str, pos: int, length: int) -> str:
    start = max(0, pos - length)
    end = min(len(text), pos + length)
    return text[start:end].replace("\n", " ").strip()
