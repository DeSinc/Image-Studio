//! Windows taskbar progress indicator via `ITaskbarList3`.
//!
//! Displays a coloured progress wipe on the application's taskbar button:
//! - Green (`TBPF_NORMAL`) for active downloads (engine, models, app updates)
//! - Yellow (`TBPF_PAUSED`) for image/video generation

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TaskbarState {
    None,
    Download,
    Generating,
}

#[cfg(windows)]
mod imp {
    use super::TaskbarState;
    use std::ffi::c_void;
    use windows_sys::core::GUID;
    use windows_sys::Win32::Foundation::{BOOL, HWND, LPARAM};
    use windows_sys::Win32::System::Com::{
        CoCreateInstance, CoInitializeEx, CLSCTX_ALL, COINIT_APARTMENTTHREADED,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcessId;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        EnumWindows, GetWindow, GetWindowThreadProcessId, IsWindowVisible, GW_OWNER,
    };

    const CLSID_TASKBAR_LIST: GUID = GUID {
        data1: 0x56FDF344,
        data2: 0xFD6D,
        data3: 0x11D0,
        data4: [0x95, 0x8A, 0x00, 0x60, 0x97, 0xC9, 0xA0, 0x90],
    };

    const IID_ITASKBAR_LIST3: GUID = GUID {
        data1: 0xEA1AFB91,
        data2: 0x9E28,
        data3: 0x4B86,
        data4: [0x90, 0xE9, 0x9E, 0x9F, 0x8A, 0x5E, 0xEF, 0xAF],
    };

    const TBPF_NOPROGRESS: i32 = 0x0;
    const TBPF_NORMAL: i32 = 0x2;
    const TBPF_PAUSED: i32 = 0x8;
    const PROGRESS_TOTAL: u64 = 1000;

    struct EnumState {
        pid: u32,
        hwnd: HWND,
    }

    unsafe extern "system" fn enum_windows_cb(hwnd: HWND, lparam: LPARAM) -> BOOL {
        unsafe {
            let state = &mut *(lparam as *mut EnumState);
            let mut win_pid: u32 = 0;
            GetWindowThreadProcessId(hwnd, &mut win_pid);
            if win_pid == state.pid
                && IsWindowVisible(hwnd) != 0
                && GetWindow(hwnd, GW_OWNER).is_null()
            {
                state.hwnd = hwnd;
                return 0;
            }
        }
        1
    }

    fn find_app_hwnd() -> Option<HWND> {
        let mut state = EnumState {
            pid: unsafe { GetCurrentProcessId() },
            hwnd: std::ptr::null_mut(),
        };
        unsafe {
            EnumWindows(Some(enum_windows_cb), &mut state as *mut EnumState as LPARAM);
        }
        if state.hwnd.is_null() {
            None
        } else {
            Some(state.hwnd)
        }
    }

    pub struct TaskbarProgress {
        hwnd: Option<HWND>,
        taskbar_list: Option<*mut c_void>,
        init_attempted: bool,
        last_state: TaskbarState,
        last_permille: u64,
    }

    impl Default for TaskbarProgress {
        fn default() -> Self {
            Self {
                hwnd: None,
                taskbar_list: None,
                init_attempted: false,
                last_state: TaskbarState::None,
                last_permille: 0,
            }
        }
    }

    impl TaskbarProgress {
        fn ensure_ready(&mut self) -> Option<(HWND, *mut c_void)> {
            if self.hwnd.is_none() {
                self.hwnd = find_app_hwnd();
            }
            let hwnd = self.hwnd?;

            if !self.init_attempted {
                self.init_attempted = true;
                unsafe {
                    let _ = CoInitializeEx(std::ptr::null(), COINIT_APARTMENTTHREADED as u32);
                    let mut obj: *mut c_void = std::ptr::null_mut();
                    let hr = CoCreateInstance(
                        &CLSID_TASKBAR_LIST,
                        std::ptr::null_mut(),
                        CLSCTX_ALL,
                        &IID_ITASKBAR_LIST3,
                        &mut obj,
                    );
                    if hr == 0 && !obj.is_null() {
                        let vtbl = *(obj as *const *const usize);
                        type HrInitFn = unsafe extern "system" fn(*mut c_void) -> i32;
                        let hr_init: HrInitFn = std::mem::transmute(*vtbl.add(3));
                        if hr_init(obj) == 0 {
                            self.taskbar_list = Some(obj);
                        } else {
                            type ReleaseFn = unsafe extern "system" fn(*mut c_void) -> u32;
                            let release: ReleaseFn = std::mem::transmute(*vtbl.add(2));
                            release(obj);
                        }
                    }
                }
            }

            self.taskbar_list.map(|ptr| (hwnd, ptr))
        }

        /// Updates the Windows taskbar button progress wipe.
        ///
        /// `progress` is clamped to `[0.0, 1.0]` and quantized to 0..=1000 so COM
        /// calls only happen when the visible state or percentage changes.
        pub fn update(&mut self, state: TaskbarState, progress: f32) {
            let permille = quantize_progress(progress);
            if state == self.last_state
                && (state == TaskbarState::None || permille == self.last_permille)
            {
                return;
            }

            let Some((hwnd, obj)) = self.ensure_ready() else {
                return;
            };

            unsafe {
                let vtbl = *(obj as *const *const usize);
                type SetProgressValueFn =
                    unsafe extern "system" fn(*mut c_void, HWND, u64, u64) -> i32;
                type SetProgressStateFn =
                    unsafe extern "system" fn(*mut c_void, HWND, i32) -> i32;

                let set_value: SetProgressValueFn = std::mem::transmute(*vtbl.add(9));
                let set_state: SetProgressStateFn = std::mem::transmute(*vtbl.add(10));

                match state {
                    TaskbarState::None => {
                        set_state(obj, hwnd, TBPF_NOPROGRESS);
                    }
                    TaskbarState::Download => {
                        set_value(obj, hwnd, permille, PROGRESS_TOTAL);
                        set_state(obj, hwnd, TBPF_NORMAL);
                    }
                    TaskbarState::Generating => {
                        // SetProgressValue implicitly switches NOPROGRESS -> NORMAL,
                        // so SetProgressState(TBPF_PAUSED) must follow it to keep yellow.
                        set_value(obj, hwnd, permille, PROGRESS_TOTAL);
                        set_state(obj, hwnd, TBPF_PAUSED);
                    }
                }
            }

            self.last_state = state;
            self.last_permille = permille;
        }
    }

    impl Drop for TaskbarProgress {
        fn drop(&mut self) {
            if let Some(obj) = self.taskbar_list.take() {
                unsafe {
                    let vtbl = *(obj as *const *const usize);
                    type ReleaseFn = unsafe extern "system" fn(*mut c_void) -> u32;
                    let release: ReleaseFn = std::mem::transmute(*vtbl.add(2));
                    release(obj);
                }
            }
        }
    }

    pub(super) fn quantize_progress(progress: f32) -> u64 {
        if !progress.is_finite() {
            return 0;
        }
        (progress.clamp(0.0, 1.0) * PROGRESS_TOTAL as f32).round() as u64
    }
}

#[cfg(not(windows))]
mod imp {
    use super::TaskbarState;

    #[derive(Default)]
    pub struct TaskbarProgress;

    impl TaskbarProgress {
        pub fn update(&mut self, _state: TaskbarState, _progress: f32) {}
    }

    #[allow(dead_code)]
    pub(super) fn quantize_progress(progress: f32) -> u64 {
        if !progress.is_finite() {
            return 0;
        }
        (progress.clamp(0.0, 1.0) * 1000.0).round() as u64
    }
}

pub use imp::TaskbarProgress;

#[cfg(test)]
mod tests {
    use super::imp::quantize_progress;

    #[test]
    fn quantize_progress_clamps_and_handles_nan() {
        assert_eq!(quantize_progress(-0.5), 0);
        assert_eq!(quantize_progress(0.0), 0);
        assert_eq!(quantize_progress(0.5), 500);
        assert_eq!(quantize_progress(1.0), 1000);
        assert_eq!(quantize_progress(1.5), 1000);
        assert_eq!(quantize_progress(f32::NAN), 0);
        assert_eq!(quantize_progress(f32::INFINITY), 0);
    }
}
