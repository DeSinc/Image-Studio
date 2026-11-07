use std::sync::{Arc, Mutex};
use std::time::Duration;
use serde::{Deserialize, Serialize};

pub const GITHUB_API_URL: &str = "https://api.github.com/repos/DeSinc/Image-Studio/releases/latest";
pub const SERVER_PRIMARY_URL: &str = "https://desinc.ax/image-studio/version.json";
pub const SERVER_FALLBACK_URL: &str = "https://desinc.my.to/image-studio/version.json";
#[allow(dead_code)]
pub const UPDATE_URL: &str = SERVER_PRIMARY_URL;

#[derive(Deserialize, Serialize, Clone, Debug, Default, PartialEq)]
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

#[derive(Deserialize, Debug)]
struct GithubRelease {
    tag_name: String,
    #[serde(default)]
    assets: Vec<GithubAsset>,
    #[serde(default)]
    body: Option<String>,
}

#[derive(Deserialize, Debug)]
struct GithubAsset {
    name: String,
    browser_download_url: String,
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

/// Parses a GitHub Release JSON string into a RemoteVersion
pub fn parse_github_release(json_text: &str) -> Result<RemoteVersion, String> {
    let release: GithubRelease = serde_json::from_str(json_text)
        .map_err(|e| format!("Failed to parse GitHub release JSON: {e}"))?;
    let version = release.tag_name.trim_start_matches('v').trim().to_string();
    let download_url = release
        .assets
        .into_iter()
        .find(|a| a.name.eq_ignore_ascii_case("ImageStudio.exe") || a.name.ends_with(".exe"))
        .map(|a| a.browser_download_url)
        .unwrap_or_else(|| {
            format!(
                "https://github.com/DeSinc/Image-Studio/releases/download/{}/ImageStudio.exe",
                release.tag_name
            )
        });
    Ok(RemoteVersion {
        version,
        download_url,
        changelog: release.body,
    })
}

/// Parses server version.json content, handling possible BOM prefixes
pub fn parse_server_version(json_text: &str) -> Result<RemoteVersion, String> {
    let clean = json_text.trim_start_matches('\u{feff}').trim();
    serde_json::from_str::<RemoteVersion>(clean)
        .map_err(|e| format!("Failed to parse server version JSON: {e}"))
}

/// Fetches the latest release from GitHub API
pub fn fetch_github_version(client: &reqwest::blocking::Client) -> Result<RemoteVersion, String> {
    let resp = client
        .get(GITHUB_API_URL)
        .header(reqwest::header::USER_AGENT, "ImageStudio-Updater")
        .header(reqwest::header::ACCEPT, "application/vnd.github.v3+json")
        .send()
        .map_err(|e| format!("GitHub request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("GitHub returned HTTP {}", resp.status()));
    }

    let text = resp.text().map_err(|e| format!("Failed to read GitHub response: {e}"))?;
    parse_github_release(&text)
}

/// Fetches version from server: tries desinc.ax first, falls back to desinc.my.to if unreachable
pub fn fetch_server_version(client: &reqwest::blocking::Client) -> Result<RemoteVersion, String> {
    // 1. Try primary server (.ax)
    if let Ok(resp) = client.get(SERVER_PRIMARY_URL).send() {
        if resp.status().is_success() {
            if let Ok(text) = resp.text() {
                if let Ok(v) = parse_server_version(&text) {
                    return Ok(v);
                }
            }
        }
    }

    // 2. If .ax failed or could not be resolved, fall back to .my.to
    let resp = client
        .get(SERVER_FALLBACK_URL)
        .send()
        .map_err(|e| format!("Could not reach server fallback: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("Server returned HTTP {}", resp.status()));
    }
    let text = resp.text().map_err(|e| format!("Failed to read server response: {e}"))?;
    parse_server_version(&text)
}

/// Selects the best candidate between GitHub and server releases.
/// Checks GitHub first, then compares with server: picking whichever is strictly newest.
/// If versions are identical, GitHub is favored as the primary distribution channel.
pub fn select_best_version(gh: Option<RemoteVersion>, srv: Option<RemoteVersion>) -> Option<RemoteVersion> {
    match (gh, srv) {
        (Some(g), Some(s)) => {
            if is_newer(&s.version, &g.version) {
                Some(s)
            } else {
                Some(g)
            }
        }
        (Some(g), None) => Some(g),
        (None, Some(s)) => Some(s),
        (None, None) => None,
    }
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

        let gh_res = fetch_github_version(&client);
        let srv_res = fetch_server_version(&client);

        let best = select_best_version(gh_res.ok(), srv_res.ok());

        match best {
            Some(info) => {
                let is_new = is_newer(&info.version, current_ver);
                let is_metered = crate::engine::is_metered_connection();
                let should_auto_dl = {
                    let mut st = state.lock().unwrap();
                    st.checked = true;
                    st.latest_version = info.version.clone();
                    st.download_url = info.download_url.clone();
                    st.changelog = info.changelog;
                    st.error = None;
                    st.update_available = is_new;
                    is_new && !st.downloading && !st.update_applied && !is_metered
                };
                if should_auto_dl {
                    download_and_install_update(state.clone(), ctx.clone());
                }
            }
            None => {
                let mut st = state.lock().unwrap();
                st.checked = true;
                st.error = Some("Could not check for updates from GitHub or mirror servers".to_string());
            }
        }
        ctx.request_repaint();
    });
}

/// Detaches and cleanly restarts the newly updated ImageStudio application
pub fn restart_application() {
    let exe = std::env::current_exe().unwrap_or_default();
    let target_exe = if exe.extension().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("old")).unwrap_or(false) {
        exe.with_extension("exe")
    } else {
        exe
    };

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_BREAKAWAY_FROM_JOB: u32 = 0x01000000;
        const DETACHED_PROCESS: u32 = 0x00000008;

        let target_str = target_exe.to_string_lossy().to_string();
        let cmd_script = format!("timeout /t 1 /nobreak > nul & start \"\" \"{}\"", target_str);
        let mut cmd = std::process::Command::new("cmd.exe");
        cmd.args(["/C", &cmd_script]);
        cmd.creation_flags(CREATE_BREAKAWAY_FROM_JOB | DETACHED_PROCESS);
        let _ = cmd.spawn();
        std::process::exit(0);
    }
    #[cfg(not(windows))]
    {
        let _ = std::process::Command::new(target_exe).spawn();
        std::process::exit(0);
    }
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
        let parsed = parse_server_version(raw_with_bom);
        assert!(parsed.is_ok());
        let info = parsed.unwrap();
        assert_eq!(info.version, "0.2.8");
    }

    #[test]
    fn test_parse_github_release() {
        let json = r#"{
            "tag_name": "v0.3.5",
            "body": "Bug fixes and improvements",
            "assets": [
                {
                    "name": "ImageStudio.exe",
                    "browser_download_url": "https://github.com/DeSinc/Image-Studio/releases/download/v0.3.5/ImageStudio.exe"
                }
            ]
        }"#;
        let parsed = parse_github_release(json).unwrap();
        assert_eq!(parsed.version, "0.3.5");
        assert_eq!(parsed.download_url, "https://github.com/DeSinc/Image-Studio/releases/download/v0.3.5/ImageStudio.exe");
        assert_eq!(parsed.changelog.as_deref(), Some("Bug fixes and improvements"));
    }

    #[test]
    fn test_select_best_version() {
        let gh_older = Some(RemoteVersion {
            version: "0.3.5".into(),
            download_url: "https://github.com/.../0.3.5.exe".into(),
            changelog: None,
        });
        let srv_newer = Some(RemoteVersion {
            version: "0.3.6".into(),
            download_url: "https://desinc.ax/.../0.3.6.exe".into(),
            changelog: None,
        });
        let gh_newer = Some(RemoteVersion {
            version: "0.3.7".into(),
            download_url: "https://github.com/.../0.3.7.exe".into(),
            changelog: None,
        });

        // Server newer than GitHub -> pick server
        let res = select_best_version(gh_older.clone(), srv_newer.clone());
        assert_eq!(res.unwrap().version, "0.3.6");

        // GitHub newer than server -> pick GitHub
        let res = select_best_version(gh_newer.clone(), srv_newer.clone());
        assert_eq!(res.unwrap().version, "0.3.7");

        // Equal version -> favor GitHub
        let srv_same = Some(RemoteVersion {
            version: "0.3.5".into(),
            download_url: "https://desinc.ax/.../0.3.5.exe".into(),
            changelog: None,
        });
        let res = select_best_version(gh_older.clone(), srv_same);
        assert_eq!(res.unwrap().download_url, "https://github.com/.../0.3.5.exe");

        // One source failed
        assert_eq!(select_best_version(gh_older.clone(), None).unwrap().version, "0.3.5");
        assert_eq!(select_best_version(None, srv_newer.clone()).unwrap().version, "0.3.6");
        assert!(select_best_version(None, None).is_none());
    }
}
