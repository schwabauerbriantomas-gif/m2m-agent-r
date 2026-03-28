//! Persistence layer — vectors, metadata (SQLite), energy state, WAL.

use ndarray::Array2;
use rusqlite::{Connection, params};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::wal::WriteAheadLog;

/// M2M Persistence manager.
pub struct M2MPersistence {
    pub storage_path: PathBuf,
    pub wal: Option<WriteAheadLog>,
    db_path: PathBuf,
}

impl M2MPersistence {
    pub fn new(storage_path: &str, enable_wal: bool) -> Result<Self, Box<dyn std::error::Error>> {
        // Path traversal prevention
        let resolved = PathBuf::from(storage_path).canonicalize()
            .unwrap_or_else(|_| PathBuf::from(storage_path));
        if storage_path.contains("..") {
            return Err("Path traversal detected in storage_path".into());
        }

        // Create directory structure
        for dir in &["data/vectors", "data/metadata", "index", "wal/checkpoint", "energy"] {
            std::fs::create_dir_all(resolved.join(dir))?;
        }

        let db_path = resolved.join("data").join("metadata").join("metadata.db");

        // Initialize SQLite
        let conn = Connection::open(&db_path)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS documents (
                 id TEXT PRIMARY KEY,
                 shard_idx INTEGER,
                 vector_idx INTEGER,
                 metadata TEXT,
                 document TEXT,
                 deleted INTEGER DEFAULT 0,
                 created_at REAL,
                 updated_at REAL
             );
             CREATE INDEX IF NOT EXISTS idx_deleted ON documents(deleted);"
        )?;
        drop(conn);

        // Initialize WAL
        let wal = if enable_wal {
            let wal_path = resolved.join("wal").join("wal.log");
            Some(WriteAheadLog::new(wal_path.to_str().unwrap_or(""), 50)?)
        } else {
            None
        };

        Ok(Self {
            storage_path: resolved,
            wal,
            db_path,
        })
    }

    fn get_conn(&self) -> Result<Connection, rusqlite::Error> {
        Connection::open(&self.db_path)
    }

    // ---- Vector Storage ----

    /// Save vectors as binary: [8 bytes rows][8 bytes cols][flat f32 data].
    pub fn save_vectors(&self, vectors: &Array2<f32>, shard_name: &str) -> std::io::Result<()> {
        let path = self.storage_path.join("data").join("vectors").join(format!("{}.bin", shard_name));
        let mut file = std::fs::File::create(path)?;
        use std::io::Write;
        file.write_all(&(vectors.nrows() as u64).to_le_bytes())?;
        file.write_all(&(vectors.ncols() as u64).to_le_bytes())?;
        for row in vectors.rows() {
            let slice = row.as_slice().unwrap_or(&[]);
            let byte_slice: &[u8] = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4) };
            file.write_all(byte_slice)?;
        }
        Ok(())
    }

    /// Load vectors from binary shard.
    pub fn load_vectors(&self, shard_name: &str) -> std::io::Result<Option<Array2<f32>>> {
        let path = self.storage_path.join("data").join("vectors").join(format!("{}.bin", shard_name));
        if !path.exists() {
            return Ok(None);
        }
        let mut file = std::fs::File::open(path)?;
        use std::io::Read;
        let mut buf8 = [0u8; 8];
        file.read_exact(&mut buf8)?;
        let rows = u64::from_le_bytes(buf8) as usize;
        file.read_exact(&mut buf8)?;
        let cols = u64::from_le_bytes(buf8) as usize;

        let mut data = vec![0.0f32; rows * cols];
        let byte_slice: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4)
        };
        file.read_exact(byte_slice)?;

        Ok(Some(ndarray::Array2::from_shape_vec((rows, cols), data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?))
    }

    /// List all available shards.
    pub fn list_shards(&self) -> Vec<String> {
        let dir = self.storage_path.join("data").join("vectors");
        std::fs::read_dir(&dir)
            .map(|entries| entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "bin"))
                .filter_map(|e| e.path().file_stem().map(|s| s.to_string_lossy().into_owned()))
                .collect()
            )
            .unwrap_or_default()
    }

    // ---- Metadata (SQLite) ----

    pub fn save_metadata(&self, doc_id: &str, shard_idx: i64, vector_idx: i64, metadata: Option<&serde_json::Value>, document: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs_f64();
        let conn = self.get_conn()?;
        conn.execute(
            "INSERT OR REPLACE INTO documents (id, shard_idx, vector_idx, metadata, document, deleted, created_at, updated_at) VALUES (?1,?2,?3,?4,?5,0,?6,?7)",
            params![doc_id, shard_idx, vector_idx, metadata.map(|m| m.to_string()), document, now, now],
        )?;
        Ok(())
    }

    pub fn get_metadata(&self, doc_id: &str) -> Result<Option<HashMap<String, serde_json::Value>>, Box<dyn std::error::Error>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare("SELECT metadata, document, shard_idx, vector_idx FROM documents WHERE id=?1 AND deleted=0")?;
        let result = stmt.query_row(params![doc_id], |row| {
            let meta_str: Option<String> = row.get(0)?;
            let doc: Option<String> = row.get(1)?;
            let shard: i64 = row.get(2)?;
            let vidx: i64 = row.get(3)?;
            Ok((meta_str, doc, shard, vidx))
        });

        match result {
            Ok((meta_str, doc, shard, vidx)) => {
                let mut map = HashMap::new();
                if let Some(ms) = meta_str {
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&ms) {
                        map.insert("metadata".into(), parsed);
                    }
                }
                map.insert("shard_idx".into(), serde_json::json!(shard));
                map.insert("vector_idx".into(), serde_json::json!(vidx));
                if let Some(d) = doc {
                    map.insert("document".into(), serde_json::json!(d));
                }
                Ok(Some(map))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn soft_delete(&self, doc_id: &str) -> Result<bool, Box<dyn std::error::Error>> {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs_f64();
        let conn = self.get_conn()?;
        let affected = conn.execute("UPDATE documents SET deleted=1, updated_at=?1 WHERE id=?2", params![now, doc_id])?;
        Ok(affected > 0)
    }

    pub fn hard_delete(&self, doc_id: &str) -> Result<bool, Box<dyn std::error::Error>> {
        let conn = self.get_conn()?;
        let affected = conn.execute("DELETE FROM documents WHERE id=?1", params![doc_id])?;
        Ok(affected > 0)
    }

    pub fn get_all_ids(&self, include_deleted: bool) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let conn = self.get_conn()?;
        let sql = if include_deleted { "SELECT id FROM documents" } else { "SELECT id FROM documents WHERE deleted=0" };
        let mut stmt = conn.prepare(sql)?;
        let ids: Vec<String> = stmt.query_map([], |row| row.get(0))?.filter_map(|r| r.ok()).collect();
        Ok(ids)
    }

    // ---- Energy State ----

    pub fn save_energy_state(&self, mu: &Array2<f32>, alpha: &[f32], kappa: &[f32]) -> std::io::Result<()> {
        let path = self.storage_path.join("energy").join("landscape.bin");
        let mut file = std::fs::File::create(path)?;
        use std::io::Write;
        // mu: [rows, cols, data...]
        file.write_all(&(mu.nrows() as u64).to_le_bytes())?;
        file.write_all(&(mu.ncols() as u64).to_le_bytes())?;
        for row in mu.rows() {
            let s = row.as_slice().unwrap_or(&[]);
            let bytes: &[u8] = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4) };
            file.write_all(bytes)?;
        }
        // alpha
        file.write_all(&(alpha.len() as u64).to_le_bytes())?;
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(alpha.as_ptr() as *const u8, alpha.len() * 4) };
        file.write_all(bytes)?;
        // kappa
        file.write_all(&(kappa.len() as u64).to_le_bytes())?;
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(kappa.as_ptr() as *const u8, kappa.len() * 4) };
        file.write_all(bytes)?;
        Ok(())
    }

    // ---- Backup ----

    pub fn backup(&self, backup_path: &str) -> Result<String, Box<dyn std::error::Error>> {
        if backup_path.contains("..") {
            return Err("Path traversal detected".into());
        }
        let dest = PathBuf::from(backup_path);
        std::fs::create_dir_all(&dest)?;
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();
        let backup_dest = dest.join(format!("m2m_backup_{}", ts));
        // Copy directory tree
        copy_dir_recursive(&self.storage_path, &backup_dest)?;
        Ok(backup_dest.to_string_lossy().into_owned())
    }

    // ---- WAL ----

    pub fn checkpoint(&self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref wal) = self.wal {
            wal.checkpoint()?;
        }
        Ok(())
    }

    pub fn close(&self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref wal) = self.wal {
            wal.close()?;
        }
        Ok(())
    }
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            std::fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}
