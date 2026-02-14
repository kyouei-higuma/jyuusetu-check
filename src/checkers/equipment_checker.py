"""
設備表の入力ミスチェック
- 番号の連続性・飛び番
- 重複番号
- 空行・未記入
"""
import re
from .base import BaseChecker, CheckResult, Severity


class EquipmentChecker(BaseChecker):
    """設備表の入力チェック"""

    @property
    def name(self) -> str:
        return "設備表チェック"

    def run(self, text: str) -> list[CheckResult]:
        results: list[CheckResult] = []
        # 番号パターン: 「1.」「1）」「(1)」「１．」など
        number_patterns = [
            r"(?:^|\n)\s*(\d+)\s*[．.)）]\s*",
            r"(?:^|\n)\s*[（(]\s*(\d+)\s*[）)]\s*",
        ]
        numbers: list[tuple[int, int]] = []  # (番号, 出現位置)
        for pat in number_patterns:
            for m in re.finditer(pat, text):
                try:
                    num = int(m.group(1))
                    numbers.append((num, m.start()))
                except (IndexError, ValueError):
                    continue
        # 番号でソートして連続性チェック
        if numbers:
            sorted_nums = sorted(set(n for n, _ in numbers))
            seen: set[int] = set()
            for n in sorted_nums:
                if n in seen:
                    results.append(CheckResult(
                        severity=Severity.WARNING,
                        category="番号重複",
                        message=f"設備番号「{n}」が重複している可能性があります",
                        detail="番号の重複がないか確認してください。",
                    ))
                seen.add(n)
            # 飛び番（1から始まる連番でない場合）
            if sorted_nums:
                min_n, max_n = min(sorted_nums), max(sorted_nums)
                expected = set(range(min_n, max_n + 1))
                missing = expected - set(sorted_nums)
                if missing:
                    missing_sorted = sorted(missing)[:5]
                    suffix = "他" if len(missing) > 5 else ""
                    results.append(CheckResult(
                        severity=Severity.INFO,
                        category="番号連続性",
                        message=f"番号の飛びがあります: {missing_sorted}{suffix}",
                        detail="意図した番号付けか確認してください。",
                    ))
        # 設備表らしいキーワード
        equipment_words = ["設備", "付属設備", "キッチン", "浴室", "トイレ", "エアコン", "給湯"]
        if not any(w in text for w in equipment_words):
            results.append(CheckResult(
                severity=Severity.INFO,
                category="書類種別",
                message="設備表として認識される語が少ないです。設備表のPDFか確認してください。",
            ))
        # 空欄
        if "（　）" in text or "－" * 5 in text or "ー" * 5 in text:
            results.append(CheckResult(
                severity=Severity.WARNING,
                category="空欄",
                message="空欄や長いハイフンが含まれています。記入漏れを確認してください。",
            ))
        return results
