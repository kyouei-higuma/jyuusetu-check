# -*- coding: utf-8 -*-
"""
ユーザー認証モジュール
- users.json を参照してログイン認証を行う
- パスワードは SHA-256 ハッシュで保存（平文保存なし）
- 初回ログイン時のパスワード変更に対応
"""

import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Optional

# users.json のパス（プロジェクトルート/data/users.json）
_USERS_FILE = Path(__file__).resolve().parent.parent / "data" / "users.json"

# ログイン保持用 Cookie トークンの署名鍵ファイル
_COOKIE_SECRET_FILE = _USERS_FILE.parent / ".auth_cookie_secret"


def _hash_password(password: str) -> str:
    """パスワードを SHA-256 でハッシュ化して返す"""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _load_users() -> dict:
    """users.json を読み込む。ファイルがない場合は空辞書を返す"""
    if not _USERS_FILE.exists():
        return {}
    try:
        with open(_USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # _comment キーは除外
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except Exception:
        return {}


def _save_users(users: dict):
    """users.json を保存する（_comment は先頭に保持）"""
    _USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    # 既存の _comment を保持
    existing = {}
    if _USERS_FILE.exists():
        try:
            with open(_USERS_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass
    comment = existing.get("_comment", "ユーザー管理ファイル。退社した社員は active を false に変更してください。")
    output = {"_comment": comment, **users}
    with open(_USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def _get_cookie_secret() -> bytes:
    """Cookie トークン署名用の秘密鍵を取得する。

    優先順位: 環境変数 AUTH_COOKIE_SECRET → data/.auth_cookie_secret ファイル（なければ生成）。
    Streamlit Cloud では再デプロイでファイルが消えるため、恒久運用には
    Secrets に AUTH_COOKIE_SECRET を設定することを推奨。
    """
    env_secret = os.environ.get("AUTH_COOKIE_SECRET", "")
    if env_secret:
        return env_secret.encode("utf-8")
    if _COOKIE_SECRET_FILE.exists():
        try:
            data = _COOKIE_SECRET_FILE.read_text(encoding="utf-8").strip()
            if data:
                return data.encode("utf-8")
        except Exception:
            pass
    new_secret = secrets.token_hex(32)
    try:
        _COOKIE_SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
        _COOKIE_SECRET_FILE.write_text(new_secret, encoding="utf-8")
    except Exception:
        pass
    return new_secret.encode("utf-8")


def _sign_token_payload(payload: str) -> str:
    return hmac.new(_get_cookie_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def create_login_token(user_id: str, days: int = 30) -> str:
    """ログイン保持用の署名付きトークンを発行する（Cookie 保存用）"""
    expiry = int(time.time()) + days * 24 * 60 * 60
    payload = f"{user_id}|{expiry}"
    return f"{payload}|{_sign_token_payload(payload)}"


def verify_login_token(token: str) -> Optional[dict]:
    """Cookie のトークンを検証し、有効ならユーザー情報を返す。

    - 署名不一致・期限切れ・ユーザー無効化済みの場合は None
    """
    if not token:
        return None
    parts = token.split("|")
    if len(parts) != 3:
        return None
    user_id, expiry_str, signature = parts
    payload = f"{user_id}|{expiry_str}"
    if not hmac.compare_digest(_sign_token_payload(payload), signature):
        return None
    try:
        if int(expiry_str) < time.time():
            return None
    except ValueError:
        return None
    users = _load_users()
    user = users.get(user_id)
    if not user or not user.get("active", True):
        return None
    return {
        "user_id": user_id,
        "name": user.get("name", user_id),
        "role": user.get("role", "staff"),
        "must_change_password": user.get("must_change_password", False),
    }


def authenticate(user_id: str, password: str) -> tuple[bool, Optional[dict], str]:
    """
    ログイン認証を行う。

    Returns:
        (success, user_info, message)
        - success: 認証成功なら True
        - user_info: 成功時はユーザー情報の dict、失敗時は None
        - message: エラーメッセージ（成功時は空文字）
    """
    users = _load_users()
    user_id = user_id.strip()

    if not user_id or not password:
        return False, None, "ユーザーIDとパスワードを入力してください。"

    if user_id not in users:
        return False, None, "ユーザーIDまたはパスワードが正しくありません。"

    user = users[user_id]

    if not user.get("active", True):
        return False, None, "このアカウントは無効です。管理者にお問い合わせください。"

    if _hash_password(password) != user.get("password_hash", ""):
        return False, None, "ユーザーIDまたはパスワードが正しくありません。"

    return True, {
        "user_id": user_id,
        "name": user.get("name", user_id),
        "role": user.get("role", "staff"),
        "must_change_password": user.get("must_change_password", False),
    }, ""


def change_password(user_id: str, old_password: str, new_password: str) -> tuple[bool, str]:
    """
    パスワードを変更する。

    Returns:
        (success, message)
    """
    users = _load_users()

    if user_id not in users:
        return False, "ユーザーが見つかりません。"

    user = users[user_id]

    # 初回変更の場合は旧パスワード確認をスキップ可能（must_change_password=True のとき）
    if not user.get("must_change_password", False):
        if _hash_password(old_password) != user.get("password_hash", ""):
            return False, "現在のパスワードが正しくありません。"

    if len(new_password) < 6:
        return False, "新しいパスワードは6文字以上にしてください。"

    users[user_id]["password_hash"] = _hash_password(new_password)
    users[user_id]["must_change_password"] = False
    _save_users(users)
    return True, "パスワードを変更しました。"


def get_all_users() -> dict:
    """全ユーザー情報を返す（管理者用）"""
    return _load_users()


def add_user(user_id: str, name: str, initial_password: str, role: str = "staff") -> tuple[bool, str]:
    """
    新しいユーザーを追加する（管理者用）。
    初回ログイン時にパスワード変更を求める。

    Returns:
        (success, message)
    """
    users = _load_users()
    user_id = user_id.strip()

    if not user_id:
        return False, "ユーザーIDを入力してください。"
    if user_id in users:
        return False, f"ユーザーID「{user_id}」は既に存在します。"
    if len(initial_password) < 6:
        return False, "初期パスワードは6文字以上にしてください。"

    users[user_id] = {
        "name": name,
        "password_hash": _hash_password(initial_password),
        "active": True,
        "must_change_password": True,
        "role": role,
    }
    _save_users(users)
    return True, f"ユーザー「{name}」を追加しました。初回ログイン時にパスワード変更が必要です。"


def set_user_active(user_id: str, active: bool) -> tuple[bool, str]:
    """
    ユーザーの有効/無効を切り替える（管理者用）。
    退社した社員を無効化する場合に使用。

    Returns:
        (success, message)
    """
    users = _load_users()

    if user_id not in users:
        return False, "ユーザーが見つかりません。"

    users[user_id]["active"] = active
    _save_users(users)
    status = "有効" if active else "無効"
    return True, f"ユーザー「{users[user_id]['name']}」を{status}にしました。"


def reset_user_password(user_id: str, temp_password: str) -> tuple[bool, str]:
    """
    管理者がユーザーのパスワードを強制リセットし、仮パスワードを設定する。
    次回ログイン時に強制的にパスワード変更画面に移動させる。

    Returns:
        (success, message)
    """
    users = _load_users()

    if user_id not in users:
        return False, "ユーザーが見つかりません。"
    if len(temp_password) < 6:
        return False, "仮パスワードは6文字以上にしてください。"

    users[user_id]["password_hash"] = _hash_password(temp_password)
    users[user_id]["must_change_password"] = True  # 次回ログイン時に強制変更
    _save_users(users)
    return True, f"ユーザー「{users[user_id]['name']}」のパスワードを仮パスワード「{temp_password}」にリセットしました。次回ログイン時に変更を求めます。"


def delete_user(user_id: str) -> tuple[bool, str]:
    """
    ユーザーを完全に削除する（管理者用）。

    Returns:
        (success, message)
    """
    users = _load_users()

    if user_id not in users:
        return False, "ユーザーが見つかりません。"

    name = users[user_id]["name"]
    del users[user_id]
    _save_users(users)
    return True, f"ユーザー「{name}」を完全に削除しました。"


def hash_password_util(password: str) -> str:
    """パスワードのハッシュ値を返す（初期設定用ユーティリティ）"""
    return _hash_password(password)
