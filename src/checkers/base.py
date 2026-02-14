"""
チェッカーの基底クラスと結果型
"""
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class Severity(str, Enum):
    """指摘の重要度"""
    ERROR = "error"      # 入力ミス・必須不備
    WARNING = "warning"  # 疑わしい・要確認
    INFO = "info"       # 参考情報


@dataclass
class CheckResult:
    """チェック結果1件"""
    severity: Severity
    category: str
    message: str
    detail: str = ""
    position: str = ""  # 該当箇所の抜粋やページ番号など


class BaseChecker(Protocol):
    """書類チェッカーの共通インターフェース"""

    @property
    def name(self) -> str:
        """チェッカー名（表示用）"""
        ...

    def run(self, text: str) -> list[CheckResult]:
        """テキストに対してチェックを実行し、結果のリストを返す"""
        ...
