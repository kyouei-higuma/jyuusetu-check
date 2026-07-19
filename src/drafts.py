# -*- coding: utf-8 -*-
"""
下書き（作業途中データ）管理モジュール
- 作業途中のデータ（抽出結果・アップロード資料）をユーザーごとに保存し、続きから再開できるようにする
- 保存ポリシー: 30日で自動削除、1ユーザーあたり最大10件（超過分は古いものから削除）
"""

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# 下書きデータの保存先（プロジェクトルート/data/drafts/）
_DRAFTS_DIR = Path(__file__).resolve().parent.parent / "data" / "drafts"

# 保存ポリシー
MAX_DRAFTS_PER_USER = 10
DRAFT_RETENTION_DAYS = 30


def _user_dir(user_id: str) -> Path:
    return _DRAFTS_DIR / user_id


def _draft_json_path(user_id: str, draft_id: str) -> Path:
    return _user_dir(user_id) / draft_id / "draft.json"


def _read_draft_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def delete_draft(user_id: str, draft_id: str) -> bool:
    """下書きをフォルダごと削除する"""
    draft_dir = _user_dir(user_id) / draft_id
    if not draft_dir.exists():
        return False
    try:
        shutil.rmtree(draft_dir)
        return True
    except Exception:
        return False


def cleanup_drafts(user_id: Optional[str] = None):
    """保存ポリシーに従い古い下書きを削除する。

    - 30日（DRAFT_RETENTION_DAYS）を超えた下書きは全ユーザー分削除
    - user_id を指定した場合、そのユーザーの下書きが上限（MAX_DRAFTS_PER_USER）を
      超えていれば、更新日時の古いものから削除
    """
    if not _DRAFTS_DIR.exists():
        return

    cutoff = time.time() - DRAFT_RETENTION_DAYS * 24 * 60 * 60

    # 期限切れの削除（全ユーザー）
    for udir in _DRAFTS_DIR.iterdir():
        if not udir.is_dir():
            continue
        for ddir in udir.iterdir():
            if not ddir.is_dir():
                continue
            meta = _read_draft_json(ddir / "draft.json")
            updated_ts = 0.0
            if meta:
                try:
                    updated_ts = datetime.strptime(
                        meta.get("updated", ""), "%Y-%m-%d %H:%M:%S"
                    ).timestamp()
                except ValueError:
                    updated_ts = 0.0
            if not updated_ts:
                try:
                    updated_ts = ddir.stat().st_mtime
                except Exception:
                    continue
            if updated_ts < cutoff:
                try:
                    shutil.rmtree(ddir)
                except Exception:
                    pass

    # ユーザーごとの件数上限
    if user_id:
        drafts = list_drafts(user_id, _skip_cleanup=True)
        for old in drafts[MAX_DRAFTS_PER_USER:]:
            delete_draft(user_id, old["id"])


def save_draft(
    user_id: str,
    payload: dict,
    draft_id: Optional[str] = None,
    files: Optional[list] = None,  # [{"name": str, "bytes": bytes}]
    label: str = "",
) -> str:
    """下書きを保存する。draft_id を指定すると既存の下書きを上書き更新する。

    Returns:
        下書きID
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not draft_id:
        draft_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    draft_dir = _user_dir(user_id) / draft_id
    draft_dir.mkdir(parents=True, exist_ok=True)

    # 既存メタの引き継ぎ（作成日時・ファイル一覧・ラベル）
    existing = _read_draft_json(draft_dir / "draft.json") or {}
    created = existing.get("created", now_str)
    saved_files = existing.get("files", [])

    # 添付ファイルの保存（指定された場合のみ上書き）
    if files:
        files_dir = draft_dir / "files"
        if files_dir.exists():
            shutil.rmtree(files_dir, ignore_errors=True)
        files_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []
        for f in files:
            try:
                (files_dir / f["name"]).write_bytes(f["bytes"])
                saved_files.append(f["name"])
            except Exception as e:
                print(f"[ERROR] 下書きファイル保存失敗: {e}")

    draft_data = {
        "id": draft_id,
        "user_id": user_id,
        "label": label or existing.get("label", ""),
        "created": created,
        "updated": now_str,
        "payload": payload,
        "files": saved_files,
    }
    with open(draft_dir / "draft.json", "w", encoding="utf-8") as f:
        json.dump(draft_data, f, ensure_ascii=False, indent=2)

    cleanup_drafts(user_id)
    return draft_id


def list_drafts(user_id: str, _skip_cleanup: bool = False) -> list[dict]:
    """ユーザーの下書き一覧（メタ情報）を更新日時の新しい順に返す"""
    if not _skip_cleanup:
        cleanup_drafts()
    udir = _user_dir(user_id)
    if not udir.exists():
        return []
    drafts = []
    for ddir in udir.iterdir():
        if not ddir.is_dir():
            continue
        meta = _read_draft_json(ddir / "draft.json")
        if meta:
            drafts.append(meta)
    # 同一秒に保存された場合でも順序が安定するよう ID をタイブレークに使う
    drafts.sort(key=lambda d: (d.get("updated", ""), d.get("id", "")), reverse=True)
    return drafts


def load_draft(user_id: str, draft_id: str) -> Optional[dict]:
    """下書きのメタ情報＋payload を読み込む"""
    return _read_draft_json(_draft_json_path(user_id, draft_id))


def load_draft_files(user_id: str, draft_id: str) -> list[dict]:
    """下書きに保存された添付ファイルを [{"name", "bytes"}] で返す"""
    meta = load_draft(user_id, draft_id)
    if not meta:
        return []
    files_dir = _user_dir(user_id) / draft_id / "files"
    result = []
    for name in meta.get("files", []):
        fpath = files_dir / name
        if fpath.exists():
            try:
                result.append({"name": name, "bytes": fpath.read_bytes()})
            except Exception as e:
                print(f"[ERROR] 下書きファイル読み込み失敗: {e}")
    return result
