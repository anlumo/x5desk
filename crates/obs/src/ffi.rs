//! Hand-written libobs 32.x declarations: only what the plugin uses. Every layout and constant here is checked
//! against the installed headers by `tests::layout`, which compiles a C program with `offsetof` / `sizeof`.
#![allow(non_camel_case_types)]
use std::ffi::{c_char, c_int, c_void};

pub enum obs_module_t {}
pub enum obs_source_t {}
pub enum obs_data_t {}
pub enum obs_properties_t {}
pub enum obs_property_t {}
pub enum gs_texture_t {}
pub enum gs_effect_t {}
pub enum gs_eparam_t {}
/// `typedef void gs_sync_t`
pub type gs_sync_t = c_void;

pub type Cb = unsafe extern "C" fn(*mut c_void);

/// `struct obs_source_info` up to `icon_type`; `obs_register_source_s` takes the size and zeroes the rest.
#[repr(C)]
pub struct SourceInfo {
    pub id: *const c_char,
    pub ty: c_int,
    pub output_flags: u32,
    pub get_name: Option<unsafe extern "C" fn(*mut c_void) -> *const c_char>,
    pub create: Option<unsafe extern "C" fn(*mut obs_data_t, *mut obs_source_t) -> *mut c_void>,
    pub destroy: Option<Cb>,
    pub get_width: Option<unsafe extern "C" fn(*mut c_void) -> u32>,
    pub get_height: Option<unsafe extern "C" fn(*mut c_void) -> u32>,
    pub get_defaults: Option<unsafe extern "C" fn(*mut obs_data_t)>,
    pub get_properties: Option<unsafe extern "C" fn(*mut c_void) -> *mut obs_properties_t>,
    pub update: Option<unsafe extern "C" fn(*mut c_void, *mut obs_data_t)>,
    pub activate: Option<Cb>,
    pub deactivate: Option<Cb>,
    pub show: Option<Cb>,
    pub hide: Option<Cb>,
    pub video_tick: Option<unsafe extern "C" fn(*mut c_void, f32)>,
    pub video_render: Option<unsafe extern "C" fn(*mut c_void, *mut gs_effect_t)>,
    /// filter_video … audio_mix
    pub _b: [Option<Cb>; 20],
    pub icon_type: c_int,
}

/// `struct obs_source_audio`
#[repr(C)]
pub struct Audio {
    pub data: [*const u8; 8],
    pub frames: u32,
    pub speakers: c_int,
    pub format: c_int,
    pub samples_per_sec: u32,
    pub timestamp: u64,
}

/// `struct vec4`
#[repr(C, align(16))]
pub struct Vec4(pub [f32; 4]);

pub const OBS_SOURCE_TYPE_INPUT: c_int = 0;
pub const OBS_SOURCE_VIDEO: u32 = 1 << 0;
pub const OBS_SOURCE_AUDIO: u32 = 1 << 1;
pub const OBS_SOURCE_CUSTOM_DRAW: u32 = 1 << 3;
pub const OBS_ICON_TYPE_CAMERA: c_int = 9;
pub const OBS_COMBO_TYPE_LIST: c_int = 2;
pub const OBS_COMBO_FORMAT_STRING: c_int = 3;
pub const OBS_TEXT_DEFAULT: c_int = 0;
pub const OBS_TEXT_INFO: c_int = 3;
pub const OBS_EFFECT_DEFAULT: c_int = 0;
pub const AUDIO_FORMAT_FLOAT: c_int = 4;
pub const GS_R8: c_int = 2;
pub const GS_RGBA: c_int = 3;
pub const GS_BGRA: c_int = 5;
pub const GS_R8G8: c_int = 18;
pub const LOG_WARNING: c_int = 200;
pub const LOG_INFO: c_int = 300;
pub const LIBOBS_API_VER: u32 = 32 << 24 | 2 << 16 | 2;

#[link(name = "obs")]
unsafe extern "C" {
    pub fn obs_register_source_s(info: *const SourceInfo, size: usize);
    pub fn blog(level: c_int, format: *const c_char, ...);
    pub fn bfree(p: *mut c_void);
    pub fn os_gettime_ns() -> u64;

    pub fn obs_data_get_string(d: *mut obs_data_t, name: *const c_char) -> *const c_char;
    pub fn obs_data_set_default_string(d: *mut obs_data_t, name: *const c_char, v: *const c_char);
    pub fn obs_properties_create() -> *mut obs_properties_t;
    pub fn obs_properties_add_list(
        p: *mut obs_properties_t,
        name: *const c_char,
        desc: *const c_char,
        ty: c_int,
        format: c_int,
    ) -> *mut obs_property_t;
    pub fn obs_property_list_add_string(
        p: *mut obs_property_t,
        name: *const c_char,
        v: *const c_char,
    ) -> usize;
    pub fn obs_properties_add_text(
        p: *mut obs_properties_t,
        name: *const c_char,
        desc: *const c_char,
        ty: c_int,
    ) -> *mut obs_property_t;
    pub fn obs_properties_add_button2(
        p: *mut obs_properties_t,
        name: *const c_char,
        text: *const c_char,
        cb: unsafe extern "C" fn(*mut obs_properties_t, *mut obs_property_t, *mut c_void) -> bool,
        data: *mut c_void,
    ) -> *mut obs_property_t;
    pub fn obs_source_output_audio(s: *mut obs_source_t, a: *const Audio);

    pub fn obs_enter_graphics();
    pub fn obs_leave_graphics();
    pub fn obs_get_base_effect(e: c_int) -> *mut gs_effect_t;
    pub fn gs_effect_create(
        src: *const c_char,
        file: *const c_char,
        err: *mut *mut c_char,
    ) -> *mut gs_effect_t;
    pub fn gs_effect_destroy(e: *mut gs_effect_t);
    pub fn gs_effect_get_param_by_name(
        e: *const gs_effect_t,
        name: *const c_char,
    ) -> *mut gs_eparam_t;
    pub fn gs_effect_set_texture(p: *mut gs_eparam_t, t: *mut gs_texture_t);
    pub fn gs_effect_set_vec4(p: *mut gs_eparam_t, v: *const Vec4);
    pub fn gs_effect_loop(e: *mut gs_effect_t, technique: *const c_char) -> bool;
    pub fn gs_draw_sprite(t: *mut gs_texture_t, flip: u32, w: u32, h: u32);
    pub fn gs_texture_destroy(t: *mut gs_texture_t);
    pub fn gs_flush();
    pub fn gs_texture_create_from_dmabuf(
        w: u32,
        h: u32,
        drm_format: u32,
        color_format: c_int,
        n_planes: u32,
        fds: *const c_int,
        strides: *const u32,
        offsets: *const u32,
        modifiers: *const u64,
    ) -> *mut gs_texture_t;
    pub fn gs_query_dmabuf_modifiers_for_format(
        drm_format: u32,
        modifiers: *mut *mut u64,
        n: *mut usize,
    ) -> bool;
    pub fn gs_query_sync_capabilities() -> bool;
    pub fn gs_sync_create() -> *mut gs_sync_t;
    pub fn gs_sync_destroy(s: *mut gs_sync_t);
    pub fn gs_sync_export_syncobj_timeline_point(s: *mut gs_sync_t, fd: c_int, point: u64) -> bool;
}
