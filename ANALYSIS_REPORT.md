# システム解析レポート

実施日: 2026年2月

## 1. プロジェクト構成の整理

### 実際に使用されているファイル

| ファイル | 役割 |
|---------|------|
| `app.py` | Streamlitメインアプリ（照合チェックUI） |
| `src/ai_extractor.py` | Gemini API呼び出し、フォームチェック、数値照合 |
| `src/pdf_reader.py` | PDF→画像変換 |
| `src/utils.py` | 画像切り抜き（証拠表示用） |

### 未使用のモジュール

| ディレクトリ/ファイル | 状態 |
|---------------------|------|
| `src/checkers/` | **未使用**。契約書・設備表チェック用の旧実装。app.pyから参照されていない |

**推奨**: `src/checkers/` は将来の拡張用に残すか、削除してプロジェクトを簡素化するか検討してください。

---

## 2. 実施した修正

### 2.1 README.md
- 実際の機能（クロスチェック）に合わせて全面更新
- Streamlit Cloud デプロイ手順を追加
- プロジェクト構成を現状に合わせて修正

### 2.2 .gitignore（新規作成）
- `__pycache__/`、`venv/` 等のPython標準
- `.streamlit/secrets.toml`（APIキー保護。既にGitに含まれている場合は `git rm --cached .streamlit/secrets.toml` で追跡解除を検討）
- IDE・OSの一時ファイル

### 2.3 requirements.txt
- `google-generativeai>=0.8.0` にバージョン指定を追加
- 未使用の `regex` パッケージを削除

---

## 3. エラーチェック結果

| 項目 | 結果 |
|------|------|
| Linterエラー | なし |
| 未使用import | なし（app.pyのloggingは使用箇所あり） |
| 例外処理 | 適切に実装済み |
| 型ヒント | 主要箇所に実装済み |

---

## 4. セーフティブロック対策の現状

| 対策 | 状態 |
|------|------|
| プロンプトの業務宣言（日英） | ✅ 実装済み |
| セーフティ設定 BLOCK_NONE | ✅ 4カテゴリ + CIVIC_INTEGRITY |
| リトライ（1回） | ✅ 実装済み |

**制限**: Gemini APIにはPII（個人識別情報）に対する**非調整可能な保護**があり、登記簿・契約書の住所・氏名でブロックされる場合があります。完全な回避はAPI側の制限により困難です。

---

## 5. 今後の改善提案

| 優先度 | 内容 |
|--------|------|
| 高 | セーフティブロック時の代替案（例: テキスト抽出のみの簡易モード、マスキングオプション）の検討 |
| 中 | `src/checkers/` の削除または統合判断 |
| 低 | ログ出力の強化（デバッグ用） |
| 低 | 単体テストの追加 |

---

## 6. ファイル依存関係

```
app.py
├── src.ai_extractor (verify_disclosure_against_evidence, JSONParseError, SafetyBlockError)
├── src.pdf_reader (pdf_to_images)
└── src.utils (crop_evidence_region)

src/ai_extractor.py
├── google.generativeai
├── _run_form_check (内部)
└── _parse_issues_json (内部)

src/pdf_reader.py
├── fitz (PyMuPDF)
└── PIL (Pillow)
```
