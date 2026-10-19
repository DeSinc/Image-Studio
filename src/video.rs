//! High-performance video decoding and playback for Image Studio.
//! Uses `openh264` and `mp4` container parsing with fallback to local ffmpeg if present.
//! Features integrated audio playback via `rodio` and symphonia/ffmpeg decoding.

use eframe::egui::{ColorImage, Context, TextureHandle, TextureOptions};
use openh264::decoder::Decoder;
use openh264::formats::YUVSource;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

pub struct AudioPlayer {
    _sink: rodio::MixerDeviceSink,
    player: rodio::Player,
    audio_bytes: Vec<u8>,
    pub is_muted: bool,
    pub volume: f32,
}

impl AudioPlayer {
    pub fn new(audio_bytes: Vec<u8>) -> Option<Self> {
        let sink = rodio::stream::DeviceSinkBuilder::open_default_sink().ok()?;
        let player = rodio::Player::connect_new(sink.mixer());
        let decoder = rodio::Decoder::new(Cursor::new(audio_bytes.clone())).ok()?;
        player.append(decoder);
        player.play();
        Some(Self {
            _sink: sink,
            player,
            audio_bytes,
            is_muted: false,
            volume: 1.0,
        })
    }

    pub fn play(&self) {
        self.player.play();
    }

    pub fn pause(&self) {
        self.player.pause();
    }

    pub fn get_pos(&self) -> Duration {
        self.player.get_pos()
    }

    pub fn seek(&mut self, pos: Duration, is_playing: bool) {
        let seek_success = if pos.is_zero() {
            false
        } else {
            self.player.try_seek(pos).is_ok()
        };

        if !seek_success {
            self.player.clear();
            if let Ok(dec) = rodio::Decoder::new(Cursor::new(self.audio_bytes.clone())) {
                self.player.append(dec);
                if !pos.is_zero() {
                    let _ = self.player.try_seek(pos);
                }
            }
        }

        if is_playing {
            self.player.play();
        } else {
            self.player.pause();
        }
    }

    pub fn restart(&mut self) {
        self.player.clear();
        if let Ok(dec) = rodio::Decoder::new(Cursor::new(self.audio_bytes.clone())) {
            self.player.append(dec);
            self.player.play();
        }
    }

    pub fn set_volume(&mut self, vol: f32) {
        self.volume = vol.clamp(0.0, 1.0);
        if self.is_muted {
            self.player.set_volume(0.0);
        } else {
            self.player.set_volume(self.volume);
        }
    }

    pub fn toggle_mute(&mut self) {
        self.is_muted = !self.is_muted;
        if self.is_muted {
            self.player.set_volume(0.0);
        } else {
            self.player.set_volume(self.volume);
        }
    }
}

pub struct VideoPlayer {
    pub frames: Vec<ColorImage>,
    pub textures: Vec<Option<TextureHandle>>,
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    pub current_frame: usize,
    pub is_playing: bool,
    pub last_frame_time: Instant,
    pub loop_playback: bool,
    pub mp4_bytes: Vec<u8>,
    pub filename: String,
    pub subfolder: String,
    pub audio: Option<AudioPlayer>,
}

impl VideoPlayer {
    pub fn from_mp4_bytes(bytes: Vec<u8>, filename: String, subfolder: String) -> Result<Self, String> {
        let (frames, fps, width, height) = decode_mp4_to_frames(&bytes)
            .or_else(|e| {
                crate::engine::log(&format!("openh264 decode failed: {e}; trying ffmpeg fallback"));
                decode_via_ffmpeg(&bytes)
            })?;

        if frames.is_empty() {
            return Err("No video frames could be decoded from MP4".into());
        }

        let audio_bytes = extract_audio_bytes(&bytes);
        let audio = audio_bytes.and_then(AudioPlayer::new);

        let num_frames = frames.len();
        Ok(Self {
            textures: vec![None; num_frames],
            frames,
            width,
            height,
            fps: if fps > 0.0 { fps } else { 25.0 },
            current_frame: 0,
            is_playing: true,
            last_frame_time: Instant::now(),
            loop_playback: true,
            mp4_bytes: bytes,
            filename,
            subfolder,
            audio,
        })
    }

    pub fn current_texture(&mut self, ctx: &Context) -> Option<&TextureHandle> {
        if self.frames.is_empty() {
            return None;
        }
        let idx = self.current_frame.min(self.frames.len() - 1);
        if self.textures[idx].is_none() {
            let label = format!("video_frame_{idx}");
            let tex = ctx.load_texture(label, self.frames[idx].clone(), TextureOptions::LINEAR);
            self.textures[idx] = Some(tex);
        }
        self.textures[idx].as_ref()
    }

    pub fn update(&mut self, ctx: &Context) {
        if self.is_playing && self.frames.len() > 1 {
            let total_dur = self.frames.len() as f32 / self.fps.max(1.0);
            let frame_dur = std::time::Duration::from_secs_f32(1.0 / self.fps.max(1.0));

            if let Some(ref mut audio) = self.audio {
                let audio_pos_sec = audio.get_pos().as_secs_f32();
                let target_frame = (audio_pos_sec * self.fps).floor() as usize;

                if target_frame >= self.frames.len() || audio_pos_sec >= total_dur {
                    if self.loop_playback {
                        self.current_frame = 0;
                        audio.restart();
                        self.last_frame_time = Instant::now();
                    } else {
                        self.current_frame = self.frames.len().saturating_sub(1);
                        self.is_playing = false;
                        audio.pause();
                    }
                } else {
                    self.current_frame = target_frame;
                }
            } else {
                let elapsed = self.last_frame_time.elapsed();
                if elapsed >= frame_dur {
                    let advance = (elapsed.as_secs_f32() / frame_dur.as_secs_f32()).floor() as usize;
                    if advance > 0 {
                        self.last_frame_time += frame_dur * (advance as u32);
                        let next_frame = self.current_frame + advance;
                        if next_frame >= self.frames.len() {
                            if self.loop_playback {
                                self.current_frame = next_frame % self.frames.len();
                            } else {
                                self.current_frame = self.frames.len().saturating_sub(1);
                                self.is_playing = false;
                            }
                        } else {
                            self.current_frame = next_frame;
                        }
                    }
                }
            }
            ctx.request_repaint_after(frame_dur);
        }
    }

    pub fn toggle_play(&mut self) {
        self.is_playing = !self.is_playing;
        if self.is_playing {
            if self.current_frame >= self.frames.len().saturating_sub(1) {
                self.current_frame = 0;
                if let Some(ref mut audio) = self.audio {
                    audio.restart();
                }
            } else if let Some(ref audio) = self.audio {
                audio.play();
            }
        } else if let Some(ref audio) = self.audio {
            audio.pause();
        }
        self.last_frame_time = Instant::now();
    }

    pub fn seek(&mut self, frame: usize) {
        if !self.frames.is_empty() {
            self.current_frame = frame.min(self.frames.len() - 1);
            let sec = self.current_frame as f32 / self.fps;
            if let Some(ref mut audio) = self.audio {
                audio.seek(Duration::from_secs_f32(sec), self.is_playing);
            }
            self.last_frame_time = Instant::now();
        }
    }
}

fn find_ffmpeg() -> Option<PathBuf> {
    let mut cmd = std::process::Command::new("ffmpeg");
    #[cfg(windows)]
    cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    if let Ok(p) = cmd.arg("-version").output() {
        if p.status.success() {
            return Some("ffmpeg".into());
        }
    }
    let candidates = [
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ];
    for c in candidates {
        let p = Path::new(c);
        if p.exists() {
            return Some(p.to_path_buf());
        }
    }
    None
}

fn extract_audio_bytes(bytes: &[u8]) -> Option<Vec<u8>> {
    // 1. Direct in-memory decoding via rodio/symphonia (AAC, MP3, etc. inside MP4 container)
    if rodio::Decoder::new(Cursor::new(bytes.to_vec())).is_ok() {
        return Some(bytes.to_vec());
    }

    // 2. Fallback to extracting 16-bit PCM WAV via ffmpeg if installed
    if let Some(ffmpeg_path) = find_ffmpeg() {
        let temp_mp4 = std::env::temp_dir().join(format!("studio_a_{}.mp4", std::process::id()));
        if std::fs::write(&temp_mp4, bytes).is_ok() {
            let mut cmd = std::process::Command::new(ffmpeg_path);
            #[cfg(windows)]
            cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
            let out_res = cmd
                .args([
                    "-y",
                    "-i",
                    temp_mp4.to_str().unwrap(),
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "44100",
                    "-ac",
                    "2",
                    "-f",
                    "wav",
                    "pipe:1",
                ])
                .output();
            let _ = std::fs::remove_file(&temp_mp4);
            if let Ok(out) = out_res {
                if out.status.success() && !out.stdout.is_empty() {
                    return Some(out.stdout);
                }
            }
        }
    }

    None
}

/// Decodes an MP4 buffer using pure `mp4` parsing + `openh264`.
fn decode_mp4_to_frames(bytes: &[u8]) -> Result<(Vec<ColorImage>, f32, u32, u32), String> {
    let size = bytes.len() as u64;
    let cursor = Cursor::new(bytes);
    let mut mp4_reader = mp4::Mp4Reader::read_header(cursor, size).map_err(|e| format!("mp4 header: {e}"))?;

    let track_id = mp4_reader
        .tracks()
        .iter()
        .find(|(_, track)| track.media_type().map(|m| m == mp4::MediaType::H264).unwrap_or(false))
        .map(|(id, _)| *id)
        .ok_or_else(|| "No H.264 track found in MP4".to_string())?;

    let (timescale, duration, sps, pps) = {
        let track = mp4_reader.tracks().get(&track_id).ok_or("Track not found")?;
        (
            track.timescale() as f32,
            track.duration().as_millis() as f32 / 1000.0,
            track.sequence_parameter_set().map(|s| s.to_vec()).ok(),
            track.picture_parameter_set().map(|p| p.to_vec()).ok(),
        )
    };
    let sample_count = mp4_reader.sample_count(track_id).map_err(|e| format!("sample count: {e}"))?;

    // Determine accurate FPS: prefer sample duration from track if available
    let fps = if let Ok(Some(sample)) = mp4_reader.read_sample(track_id, 1) {
        if sample.duration > 0 && timescale > 0.0 {
            timescale / sample.duration as f32
        } else if duration > 0.0 && sample_count > 0 {
            sample_count as f32 / duration
        } else {
            24.0
        }
    } else if duration > 0.0 && sample_count > 0 {
        sample_count as f32 / duration
    } else {
        24.0
    };

    let config = openh264::decoder::DecoderConfig::new()
        .flush_after_decode(openh264::decoder::Flush::NoFlush);
    let mut decoder = Decoder::with_api_config(openh264::OpenH264API::from_source(), config)
        .map_err(|e| format!("openh264 init: {e}"))?;

    // Initialize OpenH264 decoder with SPS and PPS sequence headers from AVCC container
    if let Some(sps_bytes) = sps {
        let mut sps_pkt = Vec::with_capacity(4 + sps_bytes.len());
        sps_pkt.extend_from_slice(&[0, 0, 0, 1]);
        sps_pkt.extend_from_slice(&sps_bytes);
        let _ = decoder.decode(&sps_pkt);
    }
    if let Some(pps_bytes) = pps {
        let mut pps_pkt = Vec::with_capacity(4 + pps_bytes.len());
        pps_pkt.extend_from_slice(&[0, 0, 0, 1]);
        pps_pkt.extend_from_slice(&pps_bytes);
        let _ = decoder.decode(&pps_pkt);
    }

    let mut frames = Vec::new();
    let mut width = 0u32;
    let mut height = 0u32;

    for sample_id in 1..=sample_count {
        if let Ok(Some(sample)) = mp4_reader.read_sample(track_id, sample_id) {
            let data = &sample.bytes;
            let mut offset = 0;
            let mut sample_pkt = Vec::with_capacity(data.len() + 64);
            while offset + 4 <= data.len() {
                let len = u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
                offset += 4;
                if offset + len > data.len() {
                    break;
                }
                let nal = &data[offset..offset + len];
                offset += len;

                // Prepend Annex-B start code [0, 0, 0, 1] for openh264
                sample_pkt.extend_from_slice(&[0, 0, 0, 1]);
                sample_pkt.extend_from_slice(nal);
            }

            if let Ok(Some(yuv)) = decoder.decode(&sample_pkt) {
                let (img, w, h) = yuv_to_color_image(&yuv);
                width = w;
                height = h;
                frames.push(img);
            }
        }
    }

    // Flush any remaining buffered frames (e.g. B-frame reordering queue)
    if let Ok(flushed) = decoder.flush_remaining() {
        for yuv in flushed {
            let (img, w, h) = yuv_to_color_image(&yuv);
            width = w;
            height = h;
            frames.push(img);
        }
    }

    if frames.is_empty() {
        return Err("No frames extracted via OpenH264".into());
    }

    Ok((frames, fps, width, height))
}

fn yuv_to_color_image(yuv: &openh264::decoder::DecodedYUV) -> (ColorImage, u32, u32) {
    let (w, h) = yuv.dimensions();
    let mut rgb = vec![0u8; w * h * 3];
    yuv.write_rgb8(&mut rgb);

    let mut rgba = Vec::with_capacity(w * h * 4);
    for chunk in rgb.chunks_exact(3) {
        rgba.push(chunk[0]);
        rgba.push(chunk[1]);
        rgba.push(chunk[2]);
        rgba.push(255);
    }
    (ColorImage::from_rgba_unmultiplied([w, h], &rgba), w as u32, h as u32)
}

/// Fallback decoder if local ffmpeg or python is present
fn decode_via_ffmpeg(bytes: &[u8]) -> Result<(Vec<ColorImage>, f32, u32, u32), String> {
    let ffmpeg_bin = find_ffmpeg().ok_or_else(|| "ffmpeg not found on this system".to_string())?;
    let temp_mp4 = std::env::temp_dir().join(format!("studio_temp_{}.mp4", std::process::id()));
    std::fs::write(&temp_mp4, bytes).map_err(|e| format!("write temp mp4: {e}"))?;

    let out_pattern = std::env::temp_dir().join(format!("studio_f_{}_%04d.png", std::process::id()));

    // Try reading true FPS from container metadata if possible
    let true_fps = {
        let cursor = Cursor::new(bytes);
        if let Ok(mut r) = mp4::Mp4Reader::read_header(cursor, bytes.len() as u64) {
            if let Some((id, t)) = r.tracks().iter().find(|(_, t)| t.media_type().map(|m| m == mp4::MediaType::H264).unwrap_or(false)) {
                let id = *id;
                let timescale = t.timescale() as f32;
                if let Ok(Some(sample)) = r.read_sample(id, 1) {
                    if sample.duration > 0 && timescale > 0.0 {
                        timescale / sample.duration as f32
                    } else {
                        24.0
                    }
                } else {
                    24.0
                }
            } else {
                24.0
            }
        } else {
            24.0
        }
    };

    let mut cmd = std::process::Command::new(ffmpeg_bin);
    #[cfg(windows)]
    cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    let cmd_res = cmd
        .args([
            "-y",
            "-i",
            temp_mp4.to_str().unwrap(),
            "-vsync",
            "0",
            out_pattern.to_str().unwrap(),
        ])
        .output();

    let _ = std::fs::remove_file(&temp_mp4);

    let output = cmd_res.map_err(|e| format!("ffmpeg execution failed: {e}"))?;
    if !output.status.success() {
        return Err(format!("ffmpeg returned error: {}", String::from_utf8_lossy(&output.stderr)));
    }

    let mut frames = Vec::new();
    let mut width = 0u32;
    let mut height = 0u32;
    let mut idx = 1;
    loop {
        let frame_path = std::env::temp_dir().join(format!("studio_f_{}_{:04}.png", std::process::id(), idx));
        if !frame_path.exists() {
            break;
        }
        if let Ok(file_bytes) = std::fs::read(&frame_path) {
            let _ = std::fs::remove_file(&frame_path);
            if let Ok(dyn_img) = image::load_from_memory(&file_bytes) {
                let rgba = dyn_img.to_rgba8();
                width = dyn_img.width();
                height = dyn_img.height();
                frames.push(ColorImage::from_rgba_unmultiplied(
                    [width as usize, height as usize],
                    rgba.as_raw(),
                ));
            }
        }
        idx += 1;
    }

    if frames.is_empty() {
        return Err("No frames extracted by ffmpeg fallback".into());
    }

    Ok((frames, true_fps, width, height))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffmpeg_detection() {
        let _ = find_ffmpeg();
    }

    #[test]
    fn test_audio_player_creation_and_controls() {
        // Generate a minimal valid 44.1kHz mono 16-bit PCM WAV (0.1s silence)
        let sample_rate = 44100u32;
        let num_samples = 4410u32;
        let data_len = num_samples * 2;
        let mut wav = Vec::with_capacity((44 + data_len) as usize);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes()); // subchunk1 size
        wav.extend_from_slice(&1u16.to_le_bytes());  // PCM
        wav.extend_from_slice(&1u16.to_le_bytes());  // mono
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
        wav.extend_from_slice(&2u16.to_le_bytes());  // block align
        wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        wav.resize(wav.len() + data_len as usize, 0);

        if let Some(mut player) = AudioPlayer::new(wav) {
            player.pause();
            player.set_volume(0.8);
            assert!((player.volume - 0.8).abs() < 0.01);
            player.toggle_mute();
            assert!(player.is_muted);
            player.toggle_mute();
            assert!(!player.is_muted);
            player.seek(Duration::from_millis(50), false);
            player.restart();
        }
    }

    #[test]
    fn test_real_mp4_decoding() {
        let path = r"C:\Users\Dean\ComfyUI-Installs\ComfyUI\ComfyUI\output\ImageStudio\video_00001_.mp4";
        if std::path::Path::new(path).exists() {
            let bytes = std::fs::read(path).unwrap();
            let res = VideoPlayer::from_mp4_bytes(bytes, "video_00001_.mp4".into(), "ImageStudio".into());
            assert!(res.is_ok(), "VideoPlayer::from_mp4_bytes failed: {:?}", res.err());
            let player = res.unwrap();
            assert_eq!(player.frames.len(), 25);
            assert_eq!(player.fps, 24.0);
            assert_eq!(player.width, 1280);
            assert_eq!(player.height, 704);
            assert!(player.audio.is_some(), "Audio should be decoded from MP4");
        }
    }
}



