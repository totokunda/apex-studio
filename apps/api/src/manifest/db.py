import sqlite3
from typing import Dict, Any, List, DefaultDict
import json
from datetime import datetime
from loguru import logger
from src.api.config import get_components_path
from tqdm import tqdm
from src.mixins.download_mixin import DownloadMixin
# import default dict from typing
from collections import defaultdict
import threading

MANIFEST_DB = None

class ManifestDB:
    def __init__(self, path: str = ".manifest.db"):
        self.path = path
        self._local = threading.local()
        self.path_to_manifests: DefaultDict[str, Dict[str, Any]] = defaultdict(lambda: {"is_downloaded": False, "manifests": []})
        
    
    @property
    def conn(self):
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.path)
        return self._local.conn
    
    def create_table(self):
        self.conn.execute("PRAGMA foreign_keys = ON")

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS groups (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                version TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                categories TEXT,
                author TEXT,
                license TEXT,
                demo_path TEXT,
                group_type TEXT,
                full_path TEXT
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS manifests (
                id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                type TEXT NOT NULL,
                version TEXT NOT NULL,
                label TEXT,
                description TEXT,
                manifest_ref TEXT,
                is_default INTEGER NOT NULL DEFAULT 0,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                group_id TEXT,
                FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS path_manifest_mapping (
                path TEXT NOT NULL,
                manifest_id TEXT NOT NULL,
                is_downloaded INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (path, manifest_id)
                )
        """)
        
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_path_mapping_path ON path_manifest_mapping(path)")
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS manifest_path_index (
            manifest_id TEXT NOT NULL,
            path TEXT NOT NULL,
            PRIMARY KEY (manifest_id, path)
        )
        """)
        
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_manifest_path_index_path ON manifest_path_index(path)")
        
        self.conn.commit()
        
    def add_group(self, data: Dict[str, Any]):
        id = data["id"]
        name = data["name"]
        type_ = data["type"]
        version = data["api_version"]
        metadata = json.dumps(data["metadata"])
        now = datetime.now().isoformat()
        categories = json.dumps(data["categories"])
        author = data["author"]
        license_ = data["license"]
        demo_path = data["demo_path"]
        group_type = data["group_type"]
        full_path = data["full_path"]
        variants = data.get("variants", [])

        self.conn.execute(
            """
            INSERT INTO groups (
                id, name, type, version, metadata, created_at, updated_at,
                categories, author, license, demo_path, group_type, full_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                type = excluded.type,
                version = excluded.version,
                metadata = excluded.metadata,
                updated_at = excluded.updated_at,
                categories = excluded.categories,
                author = excluded.author,
                license = excluded.license,
                demo_path = excluded.demo_path,
                group_type = excluded.group_type,
                full_path = excluded.full_path
            """,
            (
                id, name, type_, version, metadata, now, now,
                categories, author, license_, demo_path, group_type, full_path
            )
        )

        for variant in variants:
            self.add_manifest(data=variant, group_id=id, commit=False)

        self.conn.commit()

    def add_manifest(self, data: Dict[str, Any], group_id: str | None = None, commit: bool = True):
        description = data.get("description")
        label = data.get("label")
        manifest_ref = data.get("manifest_ref")
        is_default = int(data.get("default", False))

        manifest_data = data["manifest"] if group_id else data

        id = manifest_data["id"]
        model = manifest_data["model"]
        type_ = manifest_data["model_type"]
        version = manifest_data["version"]

        payload = json.dumps(manifest_data)
        now = datetime.now().isoformat()

        self.conn.execute(
            """
            INSERT INTO manifests (
                id, model, type, version, data, description, label,
                manifest_ref, is_default, created_at, updated_at, group_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                model = excluded.model,
                type = excluded.type,
                version = excluded.version,
                data = excluded.data,
                description = excluded.description,
                label = excluded.label,
                manifest_ref = excluded.manifest_ref,
                is_default = excluded.is_default,
                updated_at = excluded.updated_at,
                group_id = excluded.group_id
            """,
            (id, model, type_, version, payload, description, label, manifest_ref, is_default, now, now, group_id)
        )

        if commit:
            self.conn.commit()
            
    def update_manifest(self, id: str, data: Dict[str, Any]):
        self.conn.execute("UPDATE manifests SET data = ?, updated_at = ? WHERE id = ?", (json.dumps(data), datetime.now().isoformat(), id))
        self.conn.commit()
        
    def delete_manifest(self, id: str):
        self.conn.execute("DELETE FROM manifests WHERE id = ?", (id,))
        self.conn.commit()
        
    def delete_group(self, id: str):
        self.conn.execute("DELETE FROM groups WHERE id = ?", (id,))
        self.conn.commit()
        
    def get_manifest(self, id: str):
        return self._manifest_to_dict(self.conn.execute("SELECT data FROM manifests WHERE id = ?", (id,)).fetchone()[0])
    
    def _manifest_to_dict(self, manifest:str):
        return json.loads(manifest)
    

    def get_group(self, id: str):
        query = """
            SELECT
                g.id,
                g.name,
                g.type,
                g.version,
                g.metadata,
                g.created_at,
                g.updated_at,
                g.categories,
                g.author,
                g.license,
                g.demo_path,
                g.group_type,
                g.full_path,

                m.id,
                m.model,
                m.type,
                m.version,
                m.data,
                m.description,
                m.label,
                m.manifest_ref,
                m.is_default,
                m.created_at,
                m.updated_at,
                m.group_id
            FROM groups g
            LEFT JOIN manifests m ON g.id = m.group_id
            WHERE g.id = ?
               OR g.id = (
                   SELECT group_id
                   FROM manifests
                   WHERE id = ?
               )
        """
        rows = self.conn.execute(query, (id, id)).fetchall()

        if not rows:
            return None

        group = {
            "kind": "ModelGroup",
            "id": rows[0][0],
            "name": rows[0][1],
            "type": rows[0][2],
            "api_version": rows[0][3],
            "metadata": json.loads(rows[0][4]) if rows[0][4] else {},
            "categories": json.loads(rows[0][7]) if rows[0][7] else [],
            "author": rows[0][8],
            "license": rows[0][9],
            "demo_path": rows[0][10],
            "group_type": rows[0][11],
            "full_path": rows[0][12],
            "variants": [],
        }

        for row in rows:
            if row[13] is None:
                continue

            group["variants"].append({
                "id": row[13],
                "manifest": json.loads(row[17]) if row[17] else None,
                "description": row[18],
                "label": row[19],
                "manifest_ref": row[20],
                "default": bool(row[21]),
            })

        return group
    
    def get_groups(self):
        query = """
            SELECT
                g.id,
                g.name,
                g.type,
                g.version,
                g.metadata,
                g.created_at,
                g.updated_at,
                g.categories,
                g.author,
                g.license,
                g.demo_path,
                g.group_type,
                g.full_path,
                m.id,
                m.model,
                m.type,
                m.version,
                m.data,
                m.description,
                m.label,
                m.manifest_ref,
                m.is_default,
                m.created_at,
                m.updated_at,
                m.group_id
            FROM groups g
            LEFT JOIN manifests m ON g.id = m.group_id
            ORDER BY g.created_at DESC, m.created_at ASC
        """
        rows = self.conn.execute(query).fetchall()

        groups_by_id = {}
        groups = []
    
        for row in rows:
            group_id = row[0]
    
            if group_id not in groups_by_id:
                group = {
                    "kind": "ModelGroup",
                    "id": row[0],
                    "name": row[1],
                    "type": row[2],
                    "api_version": row[3],
                    "metadata": json.loads(row[4]) if row[4] else {},
                    "categories": json.loads(row[7]) if row[7] else [],
                    "author": row[8],
                    "license": row[9],
                    "demo_path": row[10],
                    "group_type": row[11],
                    "full_path": row[12],
                    "variants": [],
                }
                groups_by_id[group_id] = group
                groups.append(group)
    
            if row[13] is None:
                continue
            
            groups_by_id[group_id]["variants"].append({
                "id": row[13],
                "manifest": json.loads(row[17]) if row[17] else None,
                "description": row[18],
                "label": row[19],
                "manifest_ref": row[20],
                "default": bool(row[21]),
            })
    
        return groups

    def get_all_manifests(self):
        return [self._manifest_to_dict(row[0]) for row in self.conn.execute("SELECT data FROM manifests").fetchall()]
    
    def get_manifests_by_model(self, model: str):
        return [self._manifest_to_dict(row[0]) for row in self.conn.execute("SELECT data FROM manifests WHERE model = ?", (model,)).fetchall()]
    
    def get_manifests_by_type(self, type: str):
        return [self._manifest_to_dict(row[0]) for row in self.conn.execute("SELECT data FROM manifests WHERE type = ?", (type,)).fetchall()]
    
    def get_manifests_by_model_and_type(self, model: str, type: str):
        return [self._manifest_to_dict(row[0]) for row in self.conn.execute("SELECT data FROM manifests WHERE model = ? AND type = ?", (model, type)).fetchall()]

    def close(self):
        self.conn.close()
        
    def __del__(self):
        self.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()    
        
    def refresh_manifest(self, manifest_id: str):
        # we need to get the manifest from the database
        from src.api.manifest import get_manifest
        manifest = get_manifest(manifest_id)
        self.update_manifest(manifest_id, manifest)
        
        
    def refresh_manifests_by_path(self, path: str):
        # check if path is in path to manifests
        try:
            dl = DownloadMixin()
            dl_path = dl.is_downloaded(path, get_components_path())
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error refreshing manifests by path {path}: {e}")
            return
        
        rows = self.conn.execute(
            "SELECT manifest_id FROM path_manifest_mapping WHERE path IN (?, ?)",
            (path, dl_path)
        ).fetchall()
        
        manifest_ids = [r[0] for r in rows]

        try:
            for manifest_id in manifest_ids:
                self.refresh_manifest(manifest_id)
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error refreshing manifests by path {path}: {e}")
            
    
    def insert_path_manifest_mapping(self, path: str, manifest_id: str):
        rows = self.conn.execute(
        "INSERT OR IGNORE INTO path_manifest_mapping (path, manifest_id) VALUES (?, ?)",
        (path, manifest_id)
        )
        return rows.rowcount
  
    def update_path_manifests(self, groups: List[Dict[str, Any]]):
        dl = DownloadMixin()

        for group in tqdm(groups, desc="Updating path manifests"):
            for variant in group["variants"]:
                spec = variant["manifest"]["spec"]
                manifest_id = variant["manifest"]['id']
                components = spec.get("components", [])
                loras = spec.get("loras", [])
                
                for component in components:
                    if component.get("model_path"):
                        paths = component["model_path"]
                        for path in paths:
                            p = path["path"] if isinstance(path, dict) else path
                            downloaded_path = dl.is_downloaded(p, get_components_path())
                            
                            if downloaded_path:
                                self.insert_path_manifest_mapping(downloaded_path, manifest_id)
                            else:
                                self.insert_path_manifest_mapping(p, manifest_id)
                            
                            remote_path = path.get("remote_path")
                            if remote_path:
                                self.insert_path_manifest_mapping(remote_path, manifest_id)
                            
                    if component.get("config_path"):
                        config_path = component["config_path"]
                        downloaded_path = dl.is_downloaded(path, get_components_path())
                        
                        if downloaded_path:
                            self.insert_path_manifest_mapping(downloaded_path, manifest_id)
                        else:
                            self.insert_path_manifest_mapping(config_path, manifest_id)
                        
                        remote_path = component.get("remote_config_path")
                        if remote_path:
                            self.insert_path_manifest_mapping(remote_path, manifest_id)
                            
                for lora in loras:
                    if lora.get("source"):
                        source = lora["source"]
                        downloaded_path = dl.is_downloaded(source, get_components_path())
                        if downloaded_path:
                            self.insert_path_manifest_mapping(downloaded_path, manifest_id)
                        else:
                            self.insert_path_manifest_mapping(source, manifest_id)
                        
                        remote_path = lora.get("remote_source")
                        if remote_path:
                            self.insert_path_manifest_mapping(remote_path, manifest_id)
        self.conn.commit()
        
    def create_manifest_path_index(self):
        from src.api.manifest import _build_manifest_id_index_uncached
        index = _build_manifest_id_index_uncached()
        for manifest_id, path in index.items():
            self.conn.execute("INSERT INTO manifest_path_index (manifest_id, path) VALUES (?, ?) ON CONFLICT(manifest_id, path) DO NOTHING", (manifest_id, path))
        self.conn.commit()
        
    def get_manifest_path_index(self):
        return {manifest_id: path for manifest_id, path in self.conn.execute("SELECT manifest_id, path FROM manifest_path_index").fetchall()}

def get_manifest_db():
    global MANIFEST_DB
    if MANIFEST_DB is None:
        MANIFEST_DB = ManifestDB()
    return MANIFEST_DB

def setup_manifest_db():
    from src.api.manifest import _get_all_group_files_sync, _list_model_types_sync
    db = get_manifest_db()
    db.create_table()
    groups = _get_all_group_files_sync()
    for group in tqdm(groups, desc="Adding groups to database"):
        db.add_group(group)
    db.update_path_manifests(groups)
    db.create_manifest_path_index()
    _list_model_types_sync()
    logger.info("Manifest database setup complete")

