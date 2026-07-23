"""
お気に入り銘柄の管理モジュール。
ログイン機能はない個人用ツールという前提で、ローカルのJSONファイルに
銘柄コード・銘柄名のリストを保存するだけのシンプルな実装。
"""
from __future__ import annotations

import json
import os

FAVORITES_PATH = os.path.join(os.path.dirname(__file__), "favorites.json")


def load_favorites() -> list[dict]:
    """お気に入り一覧を読み込む。[{'code': '7203', 'name': 'トヨタ自動車'}, ...]"""
    if not os.path.exists(FAVORITES_PATH):
        return []
    try:
        with open(FAVORITES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_favorites(favorites: list[dict]) -> None:
    with open(FAVORITES_PATH, "w", encoding="utf-8") as f:
        json.dump(favorites, f, ensure_ascii=False, indent=2)


def add_favorite(code: str, name: str | None = None) -> list[dict]:
    """お気に入りに追加する（既に存在する場合は何もしない）。"""
    favorites = load_favorites()
    if not any(f["code"] == code for f in favorites):
        favorites.append({"code": code, "name": name or code})
        save_favorites(favorites)
    return favorites


def remove_favorite(code: str) -> list[dict]:
    """お気に入りから削除する。"""
    favorites = [f for f in load_favorites() if f["code"] != code]
    save_favorites(favorites)
    return favorites


def is_favorite(code: str) -> bool:
    return any(f["code"] == code for f in load_favorites())
