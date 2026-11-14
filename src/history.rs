use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub const MAX_HISTORY_BYTES: usize = 1_000_000; // 1 MB limit

#[derive(Default, Serialize, Deserialize)]
pub struct TrimHistory {
    pub trimmed_files: HashSet<String>,
    #[serde(default)]
    pub order: Vec<String>,
}

impl TrimHistory {
    fn config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|p| p.join("ImageStudio").join("trim_history.json"))
    }

    pub fn load() -> Self {
        if let Some(path) = Self::config_path() {
            if let Ok(data) = std::fs::read_to_string(path) {
                if let Ok(mut hist) = serde_json::from_str::<TrimHistory>(&data) {
                    if hist.order.is_empty() && !hist.trimmed_files.is_empty() {
                        hist.order = hist.trimmed_files.iter().cloned().collect();
                    }
                    hist.prune_to_max_size();
                    return hist;
                }
            }
        }
        Self::default()
    }

    pub fn prune_to_max_size(&mut self) {
        if self.order.is_empty() && !self.trimmed_files.is_empty() {
            self.order = self.trimmed_files.iter().cloned().collect();
        }
        let mut est_bytes: usize = self.order.iter().map(|s| s.len() + 8).sum();
        while est_bytes > MAX_HISTORY_BYTES && !self.order.is_empty() {
            let oldest = self.order.remove(0);
            self.trimmed_files.remove(&oldest);
            est_bytes = est_bytes.saturating_sub(oldest.len() + 8);
        }
    }

    pub fn save(&mut self) {
        self.prune_to_max_size();
        if let Some(path) = Self::config_path() {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Ok(mut json) = serde_json::to_string_pretty(self) {
                while json.len() > MAX_HISTORY_BYTES && !self.order.is_empty() {
                    let oldest = self.order.remove(0);
                    self.trimmed_files.remove(&oldest);
                    if let Ok(new_json) = serde_json::to_string_pretty(self) {
                        json = new_json;
                    } else {
                        break;
                    }
                }
                let _ = std::fs::write(path, json);
            }
        }
    }

    pub fn is_trimmed(&self, path: &Path) -> bool {
        let key = Self::canonical_key(path);
        self.trimmed_files.contains(&key)
    }

    pub fn mark_trimmed(&mut self, path: &Path) {
        let key = Self::canonical_key(path);
        if self.trimmed_files.insert(key.clone()) {
            self.order.push(key);
        }
        self.save();
    }

    pub fn unmark_trimmed(&mut self, path: &Path) {
        let key = Self::canonical_key(path);
        if self.trimmed_files.remove(&key) {
            self.order.retain(|k| k != &key);
        }
        self.save();
    }

    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.trimmed_files.clear();
        self.order.clear();
        self.save();
    }

    pub fn canonical_key(path: &Path) -> String {
        let p = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let s = p.to_string_lossy();
        if cfg!(windows) {
            s.to_lowercase()
        } else {
            s.into_owned()
        }
    }
}
