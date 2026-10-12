use std::sync::{Arc, Mutex};
use std::time::Duration;
use serde::{Deserialize, Serialize};

pub const UPDATE_URL: &str = "https://desinc.my.to/image-studio/version.json";

#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct RemoteVersion {
    pub version: String,
    pub download_url: String,
    #[serde(default)]
    pub changelog: Option<String>,
}

#[derive(Clone, Default)]
pub struct UpdaterState {
    pub checked: bool,
    pub update_available: bool,
    pub latest_version: String,
    pub download_url: String,
    pub changelog: Option<String>,
    pub error: Option<String>,
    pub downloading: bool,
    pub download_progress: f32,
    pub update_applied: bool,
}

/// Compares two semver-like version strings ("0.2.8" vs "0.2.7").
/// Returns true if `remote` is strictly newer than `current`.
pub fn is_newer(remote: &str, current: &str) -> bool {
    let parse_nums = |s: &str| -> Vec<u32> {
        s.trim_start_matches('v')
            .split('.')
            .filter_map(|part| part.split('-').next().and_then(|n| n.parse::<u32>().ok()))
            .collect()
    };
    let r_nums = parse_nums(remote);
    let c_nums = parse_nums(current);

    let max_len = r_nums.len().max(c_nums.len());
    for i in 0..max_len {
        let r_val = r_nums.get(i).copied().unwrap_or(0);
        let c_val = c_nums.get(i).copied().unwrap_or(0);
        if r_val > c_val {
            return true;
        } else if r_val < c_val {
            return false;
        }
    }
    false
}

pub fn check_for_updates(state: Arc<Mutex<UpdaterState>>, ctx: eframe::egui::Context) {
    std::thread::spawn(move || {
        let current_ver = env!("CARGO_PKG_VERSION");
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(6))
            .build();

        let client = match client {
            Ok(c) => c,
            Err(e) => {
                let mut st = state.lock().unwrap();
                st.checked = true;
                st.error = Some(format!("HTTP client error: {e}"));
                ctx.request_repaint();
                return;
            }
        };

        match client.get(UPDATE_URL).send() {
            Ok(resp) => {
                if resp.status().is_success() {
                    let text = match resp.text() {
                        Ok(t) => t,
                        Err(e) => {
                            let mut st = state.lock().unwrap();
                            st.checked = true;
                            st.error = Some(format!("Failed to read response body: {e}"));
                            ctx.request_repaint();
                            return;
                        }
                    };
                    let clean = text.trim_start_matches('\u{feff}').trim();
                    match serde_json::from_str::<RemoteVersion>(clean) {
                        Ok(info) => {
                            let mut st = state.lock().unwrap();
                            st.checked = true;
                            st.latest_version = info.version.clone();
                            st.download_url = info.download_url.clone();
                            st.changelog = info.changelog;
                            st.error = None;
                            st.update_available = is_newer(&info.version, current_ver);
                        }
                        Err(e) => {
                            let mut st = state.lock().unwrap();
                            st.checked = true;
                            st.error = Some(format!("Failed to parse version JSON: {e}"));
                        }
                    }
                } else {
                    let mut st = state.lock().unwrap();
                    st.checked = true;
                    st.error = Some(format!("Server returned HTTP {}", resp.status()));
                }
            }
            Err(e) => {
                let mut st = state.lock().unwrap();
                st.checked = true;
                st.error = Some(format!("Could not reach update server: {e}"));
            }
        }
        ctx.request_repaint();
    });
}

/// Downloads new ImageStudio executable and stages it for replacement
pub fn download_and_install_update(state: Arc<Mutex<UpdaterState>>, ctx: eframe::egui::Context) {
    std::thread::spawn(move || {
        let dl_url = {
            let mut st = state.lock().unwrap();
            st.downloading = true;
            st.download_progress = 0.0;
            st.error = None;
            st.download_url.clone()
        };
        ctx.request_repaint();

        let res = (|| -> Result<(), String> {
            let client = reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(300))
                .build()
                .map_err(|e| e.to_string())?;

            let mut resp = client.get(&dl_url).send().map_err(|e| e.to_string())?;
            if !resp.status().is_success() {
                return Err(format!("Download failed with status {}", resp.status()));
            }

            let total_size = resp.content_length().unwrap_or(0);
            let exe_path = std::env::current_exe().map_err(|e| e.to_string())?;
            let temp_new = exe_path.with_extension("update.tmp");

            let mut file = std::fs::File::create(&temp_new).map_err(|e| e.to_string())?;
            let mut downloaded = 0u64;
            let mut buf = [0u8; 64 * 1024];

            use std::io::{Read, Write};
            loop {
                let n = resp.read(&mut buf).map_err(|e| e.to_string())?;
                if n == 0 {
                    break;
                }
                file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
                downloaded += n as u64;

                if total_size > 0 {
                    let prog = (downloaded as f32 / total_size as f32).clamp(0.0, 1.0);
                    state.lock().unwrap().download_progress = prog;
                    ctx.request_repaint();
                }
            }
            drop(file);

            // On Windows, rename current exe to .old, and rename downloaded tmp to current exe
            let old_exe = exe_path.with_extension("old");
            let _ = std::fs::remove_file(&old_exe);
            std::fs::rename(&exe_path, &old_exe).map_err(|e| format!("Failed to move running executable: {e}"))?;
            std::fs::rename(&temp_new, &exe_path).map_err(|e| {
                // Try to restore old if failed
                let _ = std::fs::rename(&old_exe, &exe_path);
                format!("Failed to activate new executable: {e}")
            })?;

            Ok(())
        })();

        let mut st = state.lock().unwrap();
        st.downloading = false;
        match res {
            Ok(()) => {
                st.update_applied = true;
                st.update_available = false;
            }
            Err(e) => {
                st.error = Some(e);
            }
        }
        ctx.request_repaint();
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_is_newer() {
        assert!(is_newer("0.2.8", "0.2.7"));
        assert!(is_newer("v0.2.8", "0.2.7"));
        assert!(is_newer("1.0.0", "0.9.9"));
        assert!(!is_newer("0.2.7", "0.2.7"));
        assert!(!is_newer("0.2.6", "0.2.7"));
    }

    #[test]
    fn test_parse_version_with_bom() {
        let raw_with_bom = "\u{feff}{\"version\": \"0.2.8\", \"download_url\": \"https://desinc.my.to/image-studio/ImageStudio.exe\"}";
        let clean = raw_with_bom.trim_start_matches('\u{feff}').trim();
        let parsed: Result<RemoteVersion, _> = serde_json::from_str(clean);
        assert!(parsed.is_ok());
        let info = parsed.unwrap();
        assert_eq!(info.version, "0.2.8");
    }
}
