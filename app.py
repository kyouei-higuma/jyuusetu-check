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
from src.auth import (
    authenticate,
    change_password,
    get_all_users,
    add_user,
    set_user_active,
    reset_user_password,
    delete_user,
)
from src.history import save_history, load_all_history, delete_history


def _export_users_json_content() -> str:
    """users.json の現在の内容を文字列として返す（管理者用ダウンロード）"""
    users_file = Path(__file__).resolve().parent / "data" / "users.json"
    if users_file.exists():
        return users_file.read_text(encoding="utf-8")
    users = get_all_users()
    comment = "ユーザー管理ファイル。退社した社員は active を false に変更してください。"
    return json.dumps({"_comment": comment, **users}, ensure_ascii=False, indent=2)


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

# ---------- 画面表示のカスタマイズ（ロゴ・フッター等の非表示） ----------
hide_style = """
    <style>
    /* メニュー、ヘッダー、フッター、デプロイボタンを非表示にする */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    #stDeployButton {display: none !important;}
    [data-testid="stDeployButton"] {display: none !important;}
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# Streamlit Community Cloud の外枠（親DOM）にある「Hosted with Streamlit（王冠）」や「Created by（アバター）」、「Manage app」を非表示にする
from streamlit.components.v1 import html
html("""
<script>
    const hideParentElements = () => {
        const topDoc = window.top.document;
        
        // 1. Streamlitへのリンク（王冠マーク）を持つaタグを全て非表示
        topDoc.querySelectorAll('a[href*="streamlit.io"]').forEach(el => {
            el.style.setProperty('display', 'none', 'important');
        });
        
        // 2. 作成者プロフィールへのリンク（丸いアバターなど）を持つaタグを全て非表示
        topDoc.querySelectorAll('a[href*="share.streamlit.io"]').forEach(el => {
            el.style.setProperty('display', 'none', 'important');
        });

        // 3. 王冠マークやプロフィールアイコンを包んでいる、右下の固定コンテナ自体を非表示
        // それらのリンク要素から親を辿り、position: fixed になっている外枠コンテナごと非表示にします
        const badge = topDoc.querySelector('a[href*="streamlit.io"]') || topDoc.querySelector('a[href*="share.streamlit.io"]');
        if (badge) {
            let parent = badge.parentElement;
            while (parent && parent !== topDoc.body) {
                const style = window.getComputedStyle(parent);
                if (style.position === 'fixed') {
                    parent.style.setProperty('display', 'none', 'important');
                    break;
                }
                parent = parent.parentElement;
            }
        }

        // 4. "Manage app" ボタン（デプロイボタンや管理用ツールバー）を非表示にする
        // iframe内と親DOMの両方で #stDeployButton や data-testid="stDeployButton" を非表示にする
        const hideDeployButtons = (doc) => {
            doc.querySelectorAll('#stDeployButton, [data-testid="stDeployButton"]').forEach(el => {
                el.style.setProperty('display', 'none', 'important');
            });
        };
        hideDeployButtons(document);
        hideDeployButtons(topDoc);

        // 親DOM内で "Manage app" のテキストを持つ要素を検索して非表示にする
        const findAndHideByText = (doc, text) => {
            const xpath = `//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '${text.toLowerCase()}')]`;
            const result = doc.evaluate(xpath, doc, null, XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
            for (let i = 0; i < result.snapshotLength; i++) {
                const el = result.snapshotItem(i);
                let target = el;
                while (target && target !== doc.body) {
                    const tagName = target.tagName;
                    const isFixed = window.getComputedStyle(target).position === 'fixed';
                    if (tagName === 'BUTTON' || tagName === 'A' || isFixed) {
                        target.style.setProperty('display', 'none', 'important');
                        break;
                    }
                    target = target.parentElement;
                }
                if (!target) {
                    el.style.setProperty('display', 'none', 'important');
                }
            }
        };
        findAndHideByText(topDoc, 'Manage app');
    };
    
    // 即時実行と、Streamlit Cloudの遅延読み込みに対応するための定期実行
    hideParentElements();
    setInterval(hideParentElements, 500);
</script>
""", height=0, width=0)

# ── ログイン認証 ──────────────────────────────────────────────────
def _show_login_page():
    """ログイン画面を表示し、認証が通るまでアプリ本体を表示しない"""
    st.markdown("""
    <style>
    .login-container {
        max-width: 420px;
        margin: 6rem auto 0 auto;
        background: #fff;
        border-radius: 12px;
        padding: 2.5rem 2.5rem 2rem 2.5rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.10);
    }
    .login-title {
        text-align: center;
        color: #1a5276;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.3rem;
    }
    .login-sub {
        text-align: center;
        color: #888;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">📄 重説クロスチェックシステム</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-sub">社員ログイン</div>', unsafe_allow_html=True)

    with st.form("login_form"):
        user_id = st.text_input("ユーザーID", placeholder="例: tanaka")
        password = st.text_input("パスワード", type="password", placeholder="パスワードを入力")
        submitted = st.form_submit_button("ログイン", use_container_width=True)

    if submitted:
        ok, user_info, msg = authenticate(user_id, password)
        if ok:
            st.session_state["logged_in"] = True
            st.session_state["current_user"] = user_info
            st.rerun()
        else:
            st.error(msg)

    st.markdown('</div>', unsafe_allow_html=True)


def _show_change_password_page():
    """初回ログイン時のパスワード変更画面"""
    user = st.session_state["current_user"]
    st.markdown(f"### パスワード変更（{user['name']} さん）")
    st.info("初回ログインです。新しいパスワードを設定してください（6文字以上）。")

    with st.form("change_pw_form"):
        new_pw = st.text_input("新しいパスワード", type="password")
        new_pw2 = st.text_input("新しいパスワード（確認）", type="password")
        submitted = st.form_submit_button("パスワードを変更する", use_container_width=True)

    if submitted:
        if new_pw != new_pw2:
            st.error("パスワードが一致しません。")
        else:
            ok, msg = change_password(user["user_id"], "", new_pw)
            if ok:
                st.session_state["current_user"]["must_change_password"] = False
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)


# セッションステートでログイン状態を管理
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "current_user" not in st.session_state:
    st.session_state["current_user"] = None

# 未ログインならログイン画面だけ表示してここで停止
if not st.session_state["logged_in"]:
    _show_login_page()
    st.stop()

# 初回ログイン時はパスワード変更画面
if st.session_state["current_user"].get("must_change_password"):
    _show_change_password_page()
    st.stop()

# ---------- サイドバー ----------
with st.sidebar:
    # ── ログインユーザー情報 ───────────────────────────────────
    _cur = st.session_state["current_user"]
    st.markdown(f"**ログイン中:** {_cur['name']}")
    if st.button("ログアウト", use_container_width=True):
        st.session_state["logged_in"] = False
        st.session_state["current_user"] = None
        st.rerun()

    # パスワード変更
    with st.expander("🔑 パスワード変更"):
        with st.form("sidebar_pw_change"):
            _old_pw = st.text_input("現在のパスワード", type="password", key="spw_old")
            _new_pw = st.text_input("新しいパスワード（6文字以上）", type="password", key="spw_new")
            _new_pw2 = st.text_input("新しいパスワード（確認）", type="password", key="spw_new2")
            _pw_submit = st.form_submit_button("変更する")
        if _pw_submit:
            if _new_pw != _new_pw2:
                st.error("パスワードが一致しません。")
            else:
                _ok, _msg = change_password(_cur["user_id"], _old_pw, _new_pw)
                if _ok:
                    st.success(_msg)
                else:
                    st.error(_msg)

    # ── 管理者専用：ユーザー管理 ──────────────────────────────
    if _cur.get("role") == "admin":
        with st.expander("👥 ユーザー管理（管理者）"):
            st.markdown("**社員一覧**")
            _all_users = get_all_users()
            for _uid, _udata in _all_users.items():
                _active = _udata.get("active", True)
                _status = "✅ 有効" if _active else "❌ 無効"
                
                # 社員ごとの表示カード
                st.markdown(f"---")
                st.markdown(f"**{_udata.get('name', _uid)}** (`{_uid}`)  {_status}")
                
                if _uid != _cur["user_id"]:  # 自分自身は操作不可
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    # 有効化 / 無効化ボタン
                    _btn_label = "無効化" if _active else "有効化"
                    if col1.button(_btn_label, key=f"toggle_{_uid}", use_container_width=True):
                        _ok, _msg = set_user_active(_uid, not _active)
                        if _ok:
                            st.success(_msg)
                            st.rerun()
                        else:
                            st.error(_msg)
                            
                    # パスワードリセットボタン
                    with col2:
                        with st.popover("🔑 リセット", use_container_width=True):
                            st.markdown(f"**{_udata.get('name', _uid)}** のパスワードをリセットします。")
                            _temp_pw = st.text_input("仮パスワード（6文字以上）", value="123456", key=f"temp_pw_{_uid}")
                            if st.button("リセットを確定", key=f"reset_btn_{_uid}", type="primary", use_container_width=True):
                                _ok, _msg = reset_user_password(_uid, _temp_pw)
                                if _ok:
                                    st.success(_msg)
                                    st.rerun()
                                else:
                                    st.error(_msg)
                                    
                    # 削除ボタン
                    with col3:
                        with st.popover("🗑️ 削除", use_container_width=True):
                            st.markdown(f"⚠️ **{_udata.get('name', _uid)}** を完全に削除しますか？この操作は元に戻せません。")
                            if st.button("削除を確定", key=f"del_btn_{_uid}", type="primary", use_container_width=True):
                                _ok, _msg = delete_user(_uid)
                                if _ok:
                                    st.success(_msg)
                                    st.rerun()
                                else:
                                    st.error(_msg)

            st.markdown("---")
            st.markdown("**新しい社員を追加**")
            with st.form("add_user_form"):
                _new_uid = st.text_input("ユーザーID（半角英数字）", placeholder="例: tanaka")
                _new_name = st.text_input("氏名", placeholder="例: 田中 花子")
                _new_init_pw = st.text_input("初期パスワード（6文字以上）", type="password")
                _new_role = st.selectbox("権限", ["staff", "admin"])
                _add_submit = st.form_submit_button("追加する")
            if _add_submit:
                _ok, _msg = add_user(_new_uid, _new_name, _new_init_pw, _new_role)
                if _ok:
                    st.success(_msg)
                    st.rerun()
                else:
                    st.error(_msg)

            st.markdown("---")
            st.markdown("**GitHub への反映（users.json）**")
            st.caption(
                "社員の追加・変更後は、下のボタンで users.json をダウンロードし、"
                "ローカルの data/users.json を置き換えて GitHub に push してください。"
            )
            st.download_button(
                label="📥 users.json をダウンロード",
                data=_export_users_json_content().encode("utf-8"),
                file_name="users.json",
                mime="application/json",
                use_container_width=True,
                key="dl_users_json",
            )

        # ── 管理者専用：利用履歴・精度チェック ──────────────────────
        with st.expander("📊 利用履歴・精度チェック（管理者）"):
            st.markdown("**社員のシステム利用履歴**")
            _history_list = load_all_history()
            if not _history_list:
                st.info("利用履歴はまだありません。")
            else:
                for _hist in _history_list:
                    _h_id = _hist["id"]
                    _time = _hist["timestamp"]
                    _uname = _hist["user_name"]
                    _ref_files = _hist.get("reference_files", [])
                    _t_file = _hist.get("target_file", "")
                    _issues = _hist.get("issues", [])
                    
                    _error_count = sum(1 for issue in _issues if issue.get("status") == "error")
                    _warn_count = sum(1 for issue in _issues if issue.get("status") in ("warning", "suggestion"))
                    
                    # 履歴カード
                    st.markdown(f"---")
                    col1, col2 = st.columns([3, 1])
                    col1.markdown(f"📅 **{_time}** | 👤 **{_uname}**")
                    col1.markdown(f"結果: 🔴 エラー **{_error_count}**件 | 🟡 警告等 **{_warn_count}**件")
                    col1.caption(f"根拠資料: {', '.join(_ref_files)}")
                    col1.caption(f"重要事項説明書: {_t_file}")
                    
                    # 履歴の削除ボタン
                    if col2.button("履歴削除", key=f"del_hist_{_h_id}", use_container_width=True):
                        if delete_history(_h_id):
                            st.success("履歴を削除しました。")
                            st.rerun()
                            
                    # 詳細アコーディオン
                    with st.expander("🔍 照合データと実行ログの詳細"):
                        tab_data, tab_raw, tab_files = st.tabs(["📝 指摘事項一覧", "🤖 AIの生の応答 (JSON)", "📂 添付資料"])
                        
                        with tab_data:
                            if not _issues:
                                st.success("指摘事項はありませんでした。")
                            else:
                                for _idx, _issue in enumerate(_issues):
                                    _cat = _issue.get("category", "")
                                    _stat = _issue.get("status", "warning")
                                    _item = _issue.get("item", "")
                                    _ev = _issue.get("evidence", "")
                                    _tg = _issue.get("target", "")
                                    _msg = _issue.get("message", "")
                                    
                                    _icon = "🔴" if _stat == "error" else ("💡" if _stat == "suggestion" else "🟡")
                                    st.markdown(f"**{_icon} [{_cat}] {_item}**")
                                    st.write(f"指摘: {_msg}")
                                    st.caption(f"根拠資料（正）: {_ev}")
                                    st.caption(f"重要事項説明書（案）: {_tg}")
                                    if _idx < len(_issues) - 1:
                                        st.divider()
                                        
                        with tab_raw:
                            st.markdown("**AIから返された生の指摘データ (JSON):**")
                            st.json(_issues)
                            
                        with tab_files:
                            st.markdown("📄 **読み込ませた資料データ (PDF)**")
                            _saved_pdfs = _hist.get("saved_pdfs", [])
                            if not _saved_pdfs:
                                st.info("保存された資料データはありません。")
                            else:
                                # history_dir からPDFファイルを読み込んでダウンロードボタンを設置
                                _h_dir = Path(__file__).resolve().parent / "data" / "history" / "pdfs" / _h_id
                                for _pdf_name in _saved_pdfs:
                                    _pdf_path = _h_dir / _pdf_name
                                    if _pdf_path.exists():
                                        try:
                                            with open(_pdf_path, "rb") as _f:
                                                _pdf_bytes = _f.read()
                                            st.download_button(
                                                label=f"📥 {_pdf_name} をダウンロード",
                                                data=_pdf_bytes,
                                                file_name=_pdf_name,
                                                key=f"dl_pdf_{_h_id}_{_pdf_name}"
                                            )
                                        except Exception as _e:
                                            st.error(f"ファイル読み込みエラー: {_e}")
                                    else:
                                        st.warning(f"⚠️ ファイルが見つかりません: {_pdf_name}")

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

    # AIモデルの選択肢
    MODEL_OPTIONS = {
        "Gemini 3.5 Flash (推奨・標準)": "models/gemini-3.5-flash",
        "Gemini 3.1 Pro (超高精度・最高峰)": "models/gemini-3.1-pro",
        "Gemini 3.1 Flash-Lite (高速・軽量)": "models/gemini-3.1-flash-lite",
        "Gemini 2.5 Pro (前世代・高精度)": "models/gemini-2.5-pro",
        "Gemini 2.5 Flash (前世代・標準)": "models/gemini-2.5-flash",
    }

    # デフォルトの選択肢を Secrets 等から決定
    default_model_code = "models/gemini-3.5-flash"
    try:
        default_model_code = st.secrets.get("GEMINI_MODEL", "models/gemini-3.5-flash")
    except (AttributeError, KeyError, FileNotFoundError):
        pass

    # 存在しないモデル名が指定されていた場合の安全対策
    if default_model_code not in MODEL_OPTIONS.values():
        default_model_code = "models/gemini-3.5-flash"

    default_index = list(MODEL_OPTIONS.values()).index(default_model_code)

    selected_model_name = st.selectbox(
        "AIモデルの選択",
        options=list(MODEL_OPTIONS.keys()),
        index=default_index,
        help="書類の解析に使用するAIモデルを選択します。\n\n"
             "• Gemini 3.5 Flash: 最新世代の標準モデル。速度、コスト、精度のバランスが最も優れています。\n"
             "• Gemini 3.1 Pro: 最新世代の最上位モデル。極めて高い推論能力を持ち、緻密な照合が可能です。\n"
             "• Gemini 3.1 Flash-Lite: 高速処理と軽量化に特化したモデルです。\n"
             "• Gemini 2.5 Pro / Flash: 前世代のモデルです。",
    )
    gemini_model = MODEL_OPTIONS[selected_model_name]

    st.divider()

    # セーフティブロック対策：画像量を制限するオプション
    use_light_mode = st.checkbox(
        "簡易モード（画像量を制限）",
        value=False,
        help="資料が多いとセーフティブロックされやすい場合に有効。根拠資料・重説それぞれ最大5ページまで送信します。",
        key="light_mode",
    )

    st.divider()

    st.caption("※ PDFは画像としてGeminiで解析します。スキャンPDFも利用できます。")
    st.caption("※ AIモデルはいつでも切り替え可能です。")

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
    analyzed_pdfs = []
    try:
        for ref_file in reference_files:
            content = ref_file.read()
            analyzed_pdfs.append({"name": ref_file.name, "bytes": content})
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
        analyzed_pdfs.append({"name": target_file.name, "bytes": content})
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

    # 簡易モード：画像量を制限してセーフティブロックを回避
    max_pages = 5 if st.session_state.get("light_mode", False) else 999
    reference_images_all = reference_images_all[:max_pages]
    target_images_all = target_images_all[:max_pages]
    if st.session_state.get("light_mode", False):
        st.info("📌 簡易モード：根拠資料・重説それぞれ最大5ページまで送信しています。")

    # 使用モデルの確定（サイドバーで選択されたモデルを使用）
    if "gemini_model" not in locals() and "gemini_model" not in globals():
        gemini_model = "models/gemini-3.5-flash"

    # Geminiで照合チェック（フォームチェック → 添付資料・数値照合の2段階）
    with st.spinner("フォームチェックと照合を実行中..."):
        issues = None
        for attempt in range(2):  # セーフティブロック時は1回リトライ
            try:
                issues = verify_disclosure_against_evidence(
                    gemini_api_key, reference_images_all, target_images_all, model_name=gemini_model
                )
                
                # 履歴を保存
                try:
                    cur_user = st.session_state.get("current_user", {"user_id": "unknown", "name": "不明"})
                    save_history(
                        user_id=cur_user["user_id"],
                        user_name=cur_user["name"],
                        reference_file_names=[f.name for f in reference_files],
                        target_file_name=target_file.name,
                        issues=issues,
                        analyzed_pdfs=analyzed_pdfs
                    )
                except Exception as hist_err:
                    logging.error(f"履歴の保存に失敗しました: {hist_err}")
                
                break
            except SafetyBlockError:
                if attempt == 0:
                    st.warning("再試行中...")
                    continue
                st.error("安全性の制限により解析が中断されました。")
                st.info("💡 **対処法:** 再度お試しください。システムは自動で gemini-2.5-flash-lite にも切り替えて再試行しています。それでもブロックされる場合は、資料の量を減らすか、数分待ってから再実行してください。")
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

        # ── 照合データと実行ログの詳細（チェック実行直後の表示用） ──
        st.divider()
        with st.expander("🔍 照合データと実行ログの詳細"):
            tab_raw, tab_files = st.tabs(["🤖 AIの生の応答 (JSON)", "📂 添付資料"])
            
            with tab_raw:
                st.markdown("**AIから返された生の指摘データ (JSON):**")
                st.json(issues)
                
            with tab_files:
                st.markdown("📄 **読み込ませた資料データ (PDF)**")
                for ref_file in reference_files:
                    st.download_button(
                        label=f"📥 {ref_file.name} をダウンロード",
                        data=ref_file.getvalue(),
                        file_name=ref_file.name,
                        key=f"dl_current_ref_{ref_file.name}"
                    )
                st.download_button(
                    label=f"📥 {target_file.name} をダウンロード",
                    data=target_file.getvalue(),
                    file_name=target_file.name,
                    key=f"dl_current_target_{target_file.name}"
                )

    # 処理完了後、フラグをリセット
    st.session_state["process_started"] = False
elif not reference_files or not target_file:
    st.info("👆 上記の2つのエリアにファイルをアップロードして、「チェック開始」ボタンを押してください。")
