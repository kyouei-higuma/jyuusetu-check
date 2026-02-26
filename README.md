# 不動産重要事項説明書 クロスチェック

根拠資料（登記簿・公図など）と重要事項説明書を照合し、記載内容の一致を厳密にチェックするStreamlitアプリです。PDFは画像化してGoogle Geminiで視覚的に読み取ります。

## 機能

- **2種類のPDFアップロード**
  - **根拠資料（正）**: 登記簿、公図、地積測量図、評価証明書など
  - **重要事項説明書（案）**: チェック対象の重要事項説明書
- **2段階チェック**
  1. **フォームチェック**: 宅地建物取引士名・登録番号、弊社情報、供託所、売る主、地目、容積率、敷地道路関係図など
  2. **数値照合**: 所在・地番・地積・所有者・法令上の制限など
- **結果表示**: エラー / 警告 / アドバイスに分けて表示し、該当箇所の画像を表示

## 必要な環境

- Python 3.10 以上
- Google Gemini API キー（無料枠: gemini-2.5-flash）
- 依存パッケージは `requirements.txt` を参照

## セットアップと実行

```bash
# 仮想環境の作成（推奨）
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS / Linux

# 依存関係のインストール
pip install -r requirements.txt

# アプリの起動
streamlit run app.py
```

ブラウザで `http://localhost:8501` が開きます。

## Streamlit Cloud デプロイ

1. GitHub にリポジトリを push
2. [Streamlit Cloud](https://share.streamlit.io/) でリポジトリを接続
3. Secrets に `GOOGLE_API_KEY` または `GEMINI_API_KEY` を設定
4. オプション: `GEMINI_MODEL` でモデルを変更（デフォルト: gemini-2.5-flash）

## プロジェクト構成

```
不動産書類チェックシステムProject/
├── app.py                 # Streamlit メインアプリ
├── requirements.txt
├── README.md
├── .streamlit/
│   └── secrets.toml       # ローカル用APIキー（.gitignore推奨）
└── src/
    ├── ai_extractor.py    # Gemini で画像解析・照合（フォームチェック＋数値照合）
    ├── pdf_reader.py      # PDF を画像化（JPEG）
    └── utils.py           # 画像切り抜き等
```

## 注意事項

- PDFは全ページを画像(JPEG)に変換し、Geminiに送信します。スキャンPDFも利用できます。
- 登記簿・契約書の住所・氏名等でセーフティブロックされる場合があります。その場合は資料を減らすか、時間をおいて再試行してください。
- チェック結果は自動判定のため、最終的な確認は人が行ってください。
