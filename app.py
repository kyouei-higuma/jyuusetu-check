"""
不動産重要事項説明書 照合チェック - Streamlitアプリ
根拠資料（登記簿・公図など）と重要事項説明書を照合し、記載内容の一致をチェックします。
PDFは画像化してGoogle Gemini に視覚的に読み取らせます。
"""
import base64
import io
import json
import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加（Streamlit実行時のモジュール解決用）
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from PIL import Image

from src.ai_extractor import JSONParseError, SafetyBlockError, verify_disclosure_against_evidence
from src.pdf_reader import pdf_to_images
from src.utils import crop_evidence_region


def _normalize_box_2d(box_2d):  # noqa: ANN201
    """AIが返した box_2d を数値リスト [ymin, xmin, ymax, xmax] に統一する。文字列の場合はパースする。"""
    if box_2d is None:
        return None
    if isinstance(box_2d, str):
        try:
            box_2d = json.loads(box_2d.strip())
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(box_2d, list) or len(box_2d) != 4:
        return None
    try:
        return [float(x) for x in box_2d]
    except (TypeError, ValueError):
        return None


st.set_page_config(
    page_title="不動産書類チェック",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- サイドバー ----------
with st.sidebar:
    st.header("設定")

    # Streamlit Secrets から API キーを優先取得（Streamlit Cloud デプロイ対応）
    # GOOGLE_API_KEY: Streamlit Cloud の Secrets で一般的なキー名
    # GEMINI_API_KEY: ローカル .streamlit/secrets.toml との互換用
    gemini_api_key = ""
    try:
        gemini_api_key = st.secrets.get("GOOGLE_API_KEY", "") or st.secrets.get("GEMINI_API_KEY", "")
    except (AttributeError, KeyError, FileNotFoundError):
        pass

    # Secrets に設定がない場合のみ入力欄を表示
    if not (gemini_api_key and gemini_api_key.strip()):
        gemini_api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            key="gemini_api_key_input",
            placeholder="Google Gemini APIキーを入力",
            help="Google Gemini APIキーを入力してください。または Streamlit Secrets（GOOGLE_API_KEY / GEMINI_API_KEY）に設定してください。",
        )
    else:
        st.success("✅ APIキーは Secrets から読み込まれました")

    st.divider()

    st.caption("※ PDFは画像としてGeminiで解析します。スキャンPDFも利用できます。")
    st.caption("※ デフォルトは gemini-2.0-flash（無料枠あり）。Secrets の GEMINI_MODEL で変更可。")

# ---------- メインエリア ----------
st.title("📄 重要事項説明書 クロスチェック")
st.caption("根拠資料（登記簿・公図など）と重要事項説明書を照合し、記載内容の一致を厳密にチェックします。")

if not (gemini_api_key and gemini_api_key.strip()):
    st.warning(
        "⚠️ **APIキーが設定されていません。** "
        "左のサイドバーでGoogle Gemini APIキーを入力するか、"
        "Streamlit Cloud の Secrets に `GOOGLE_API_KEY` を設定してください。"
    )
    st.stop()

# 2カラムレイアウトでファイルアップロード
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. 根拠資料（正）")
    reference_files = st.file_uploader(
        "根拠資料（登記簿・公図・評価証明など）",
        type=["pdf"],
        accept_multiple_files=True,
        help="登記簿、公図、測量図、評価証明書などの根拠資料をアップロードしてください。複数ファイル可。",
        key="reference_files",
    )

with col2:
    st.subheader("2. 重要事項説明書（案）")
    target_file = st.file_uploader(
        "重要事項説明書",
        type=["pdf"],
        accept_multiple_files=False,
        help="チェック対象となる重要事項説明書をアップロードしてください。",
        key="target_file",
    )

# 両方アップロードされたら「チェック開始」ボタンを表示
if reference_files and target_file:
    if st.button("🔍 チェック開始", type="primary", use_container_width=True):
        # セッション状態に保存して処理を開始
        st.session_state["process_started"] = True
        st.rerun()

# 処理開始フラグが立っている場合のみ処理を実行
if st.session_state.get("process_started", False):
    # 根拠資料の画像化
    reference_images_all = []
    try:
        for ref_file in reference_files:
            content = ref_file.read()
            images_b64 = pdf_to_images(io.BytesIO(content))
            pil_images = [
                Image.open(io.BytesIO(base64.b64decode(b64))) for b64 in images_b64
            ]
            reference_images_all.extend(pil_images)
    except Exception as e:
        st.error(f"根拠資料のPDF読み込みに失敗しました: {e}")
        st.stop()

    # 重要事項説明書の画像化
    target_images_all = []
    try:
        content = target_file.read()
        images_b64 = pdf_to_images(io.BytesIO(content))
        target_images_all = [
            Image.open(io.BytesIO(base64.b64decode(b64))) for b64 in images_b64
        ]
    except Exception as e:
        st.error(f"重要事項説明書のPDF読み込みに失敗しました: {e}")
        st.stop()

    if not reference_images_all:
        st.warning("根拠資料から画像を取得できませんでした。")
        st.stop()

    if not target_images_all:
        st.warning("重要事項説明書から画像を取得できませんでした。")
        st.stop()

    # 使用モデル（Secrets の GEMINI_MODEL で上書き可。gemini-3-pro は無料枠なしのため 429 回避でフォールバック）
    try:
        gemini_model = st.secrets.get("GEMINI_MODEL", "models/gemini-2.0-flash")
    except (AttributeError, KeyError, FileNotFoundError):
        gemini_model = "models/gemini-2.0-flash"
    if "gemini-3" in str(gemini_model).lower():
        gemini_model = "models/gemini-2.0-flash"  # 無料枠なしモデルは 429 になるため強制フォールバック

    # Geminiで照合チェック（フォームチェック → 添付資料・数値照合の2段階）
    with st.spinner("フォームチェックと照合を実行中..."):
        try:
            issues = verify_disclosure_against_evidence(
                gemini_api_key, reference_images_all, target_images_all, model_name=gemini_model
            )
        except SafetyBlockError as e:
            st.error("安全性の制限により解析が中断されました。")
            st.info("💡 **対処法:** プロンプトを見直すか、再度お試しください。登記簿・契約書の住所・氏名等でブロックされる場合は、資料を分割するか2段階チェック（添付資料チェック→数値照合）の利用を検討してください。")
            st.stop()
        except JSONParseError as e:
            st.error("AIからの応答が解析できませんでした。")
            st.info("💡 **対処法:** 解析を再試行するか、資料の量を減らして再度お試しください。応答が長いと末尾が欠けることがあります。")
            with st.expander("技術詳細（生の応答を確認）"):
                st.text(e.raw_response[:10000] + ("…" if len(e.raw_response) > 10000 else ""))
            st.stop()
        except json.JSONDecodeError as e:
            st.error("Geminiの応答のJSON解析に失敗しました。")
            st.info("💡 **対処法:** 解析を再試行するか、資料の量を減らして再度お試しください。")
            st.stop()
        except Exception as e:
            st.error(f"Geminiによる解析に失敗しました。{e}")
            st.info("💡 **対処法:** 解析を再試行するか、資料の量を減らして再度お試しください。")
            st.stop()

    # 結果表示
    st.subheader("照合結果")

    if not issues:
        st.success("✅ 指摘事項はありませんでした。根拠資料と重要事項説明書の記載は一致しています。")
    else:
        error_count = sum(1 for issue in issues if issue.get("status") == "error")
        warn_count = sum(1 for issue in issues if issue.get("status") in ("warning", "suggestion"))

        col1, col2 = st.columns(2)
        col1.metric("エラー（不一致）", error_count)
        col2.metric("警告・アドバイス", warn_count)

        # 証拠画像用: Geminiに渡した順と同じ（根拠資料＋重要事項説明書）
        all_images = reference_images_all + target_images_all

        for issue in issues:
            category = issue.get("category", "")
            status = issue.get("status", "warning")
            item = issue.get("item", "")
            evidence = issue.get("evidence", "")
            target = issue.get("target", "")
            message = issue.get("message", "")
            box_2d_raw = issue.get("box_2d")
            box_2d = _normalize_box_2d(box_2d_raw)  # 文字列 "[10,20,30,40]" も数値リストに変換
            image_index = issue.get("image_index")
            if isinstance(image_index, (int, float)):
                image_index = int(image_index)

            if status == "error":
                icon = "🔴"
                color = "red"
            elif status == "suggestion":
                icon = "💡"
                color = "blue"
            else:
                icon = "🟡"
                color = "orange"

            with st.expander(f"{icon} [{category}] {item}: {message}", expanded=(status == "error")):
                col_text, col_img = st.columns([1, 1.2])

                with col_text:
                    st.write("**根拠資料（正）の記載:**", evidence)
                    st.write("**重要事項説明書（案）の記載:**", target)
                    st.caption(f"カテゴリ: {category} | 重要度: {status}")

                with col_img:
                    # image_index が有効範囲か確認（根拠資料＋重説のリストと紐付け）
                    if image_index is None or not (0 <= image_index < len(all_images)):
                        logging.warning(
                            "画像インデックスが見つかりません: image_index=%s, 画像数=%d",
                            image_index,
                            len(all_images),
                        )
                        if "box_2d" in issue or "image_index" in issue:
                            st.caption("⚠️ 画像インデックスが見つかりません（表示スキップ）")
                        continue

                    source_img = all_images[image_index]
                    # 座標がある場合は必ず画像を表示（切り抜き成功時はクロップ、失敗時は元画像をフォールバック）
                    if box_2d is not None:
                        try:
                            cropped_img = crop_evidence_region(source_img, box_2d)
                            cw, ch = cropped_img.size
                            min_height = 180
                            if ch > 0 and ch < min_height and cw > 0:
                                scale = min_height / ch
                                new_w = int(cw * scale)
                                cropped_img = cropped_img.resize((new_w, min_height), Image.Resampling.LANCZOS)
                            st.image(cropped_img, caption="指摘箇所の画像", use_container_width=True)
                        except Exception:
                            st.caption("切り抜き失敗（元画像を表示）")
                            st.image(source_img, use_container_width=True)
                    else:
                        # box_2d が無い／パースできなかった場合も元画像を小さく表示
                        st.image(source_img, caption="指摘箇所の画像（座標なし）", use_container_width=True)

    # 処理完了後、フラグをリセット
    st.session_state["process_started"] = False
elif not reference_files or not target_file:
    st.info("👆 上記の2つのエリアにファイルをアップロードして、「チェック開始」ボタンを押してください。")
