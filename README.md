# 不動産売買書類 入力ミスチェック

契約書・重要事項説明書・設備表などのPDFを読み込み、入力ミスをチェックするStreamlitアプリです。

## 機能

- **PDFアップロード**: 不動産売買に関連するPDFを1つアップロード（PDFを画像化し Google Gemini で視覚的に読み取り。スキャンPDFも対応）
- **書類種別**: 契約書 / 重要事項説明書 / 設備表 のいずれかを選択、または「自動」で全チェック実行
- **チェック内容**
  - **契約書**: 金額の桁・表記、日付の妥当性、空欄・未記入の検出
  - **重要事項説明書**: 必須項目の有無、日付・空欄のチェック
  - **設備表**: 番号の連続性・重複、空欄の検出
- **結果表示**: エラー / 警告 / 情報 に分けて表示し、該当箇所を確認可能

## 必要な環境

- Python 3.10 以上
- Google Gemini API キー（Gemini 2.0 Flash Experimental による画像解析用）
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

ブラウザで `http://localhost:8501` が開きます。PDFをアップロードしてチェックを実行してください。

## プロジェクト構成

```
不動産書類チェックシステムProject/
├── app.py                 # Streamlit メインアプリ
├── requirements.txt
├── README.md
└── src/
    ├── ai_extractor.py    # Gemini で画像→テキスト抽出（JSON形式）
    ├── pdf_reader.py      # PDF を画像化（Base64 JPEG リスト）
    └── checkers/
        ├── base.py        # チェック結果型・基底
        ├── contract_checker.py   # 契約書チェック
        ├── disclosure_checker.py # 重要事項説明書チェック
        └── equipment_checker.py   # 設備表チェック
```

## 注意事項

- **PDFの読み取り**: PDFは全ページを画像(JPEG)に変換し、Google Gemini 2.0 Flash (Experimental) に送信してテキストを抽出します。スキャンPDFも利用できます。
- サイドバーで Google Gemini API キーを入力してください。
- チェック結果は自動判定のため、最終的な確認は人が行ってください。
