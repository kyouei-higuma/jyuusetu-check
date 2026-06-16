# -*- coding: utf-8 -*-
"""
利用履歴管理モジュール
- 社員が実行した照合チェックの履歴（誰が、いつ、どのファイルを、どういう結果だったか）を保存する
- 管理者がチェック結果の履歴確認や、誤判定・エラー原因究明を行えるようにする
"""

import json
from datetime import datetime
from pathlib import Path

# 履歴データの保存先（プロジェクトルート/data/history/）
_HISTORY_DIR = Path(__file__).resolve().parent.parent / "data" / "history"


def save_history(
    user_id: str,
    user_name: str,
    reference_file_names: list[str],
    target_file_name: str,
    issues: list,
    analyzed_pdfs: list = None,  # [{"name": str, "bytes": bytes}]
) -> str:
    """
    照合チェックの実行履歴を個別のJSONファイルとして保存する。
    同時に、アップロードされたPDF資料も履歴フォルダにコピー保存する。
    
    Returns:
        保存された履歴のID（ファイル名）
    """
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_id = f"{timestamp}_{user_id}"
    file_path = _HISTORY_DIR / f"{history_id}.json"
    
    # 添付PDF資料を保存するディレクトリ
    pdf_save_dir = _HISTORY_DIR / "pdfs" / history_id
    saved_pdf_names = []
    
    if analyzed_pdfs:
        pdf_save_dir.mkdir(parents=True, exist_ok=True)
        for pdf in analyzed_pdfs:
            try:
                pdf_name = pdf["name"]
                pdf_bytes = pdf["bytes"]
                pdf_path = pdf_save_dir / pdf_name
                pdf_path.write_bytes(pdf_bytes)
                saved_pdf_names.append(pdf_name)
            except Exception as e:
                print(f"[ERROR] 履歴用PDF保存失敗: {e}")
    
    history_data = {
        "id": history_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user_id,
        "user_name": user_name,
        "reference_files": reference_file_names,
        "target_file": target_file_name,
        "issues": issues if issues is not None else [],
        "saved_pdfs": saved_pdf_names,
    }
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)
        
    return history_id


def load_all_history() -> list[dict]:
    """すべての履歴を読み込み、新しい順にソートして返す"""
    if not _HISTORY_DIR.exists():
        return []
        
    history_list = []
    for file_path in _HISTORY_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                history_list.append(data)
        except Exception as e:
            print(f"[ERROR] 履歴ファイルの読み込み失敗: {file_path.name} - {e}")
            
    # タイムスタンプの新しい順（降順）にソート
    history_list.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return history_list


def delete_history(history_id: str) -> bool:
    """特定の履歴を削除する（関連するPDFも削除）"""
    file_path = _HISTORY_DIR / f"{history_id}.json"
    
    # 関連PDFの削除
    pdf_dir = _HISTORY_DIR / "pdfs" / history_id
    if pdf_dir.exists():
        try:
            import shutil
            shutil.rmtree(pdf_dir)
        except Exception:
            pass

    if file_path.exists():
        try:
            file_path.unlink()
            return True
        except Exception:
            return False
    return False
