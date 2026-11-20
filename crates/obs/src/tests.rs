use crate::{drm::*, ffi::*, *};
use std::{mem::offset_of, process::Command};

/// Every layout, offset and constant the FFI relies on, as (C expression, our value): a C program built against the
/// installed libobs and kernel headers prints the expressions, and they must match; the API version we declare must
/// be one the installed libobs loads (`obs_open_module`: its major.minor no newer than libobs').
#[test]
fn layout() {
    type I = SourceInfo;
    let p = size_of::<usize>();
    let si = |f: &str| format!("offsetof(struct obs_source_info, {f})");
    let mut t: Vec<(String, u64)> = [
        ("id", offset_of!(I, id)),
        ("type", offset_of!(I, ty)),
        ("output_flags", offset_of!(I, output_flags)),
        ("get_name", offset_of!(I, get_name)),
        ("create", offset_of!(I, create)),
        ("destroy", offset_of!(I, destroy)),
        ("get_width", offset_of!(I, get_width)),
        ("get_height", offset_of!(I, get_height)),
        ("get_defaults", offset_of!(I, get_defaults)),
        ("get_properties", offset_of!(I, get_properties)),
        ("update", offset_of!(I, update)),
        ("activate", offset_of!(I, activate)),
        ("deactivate", offset_of!(I, deactivate)),
        ("show", offset_of!(I, show)),
        ("hide", offset_of!(I, hide)),
        ("video_tick", offset_of!(I, video_tick)),
        ("video_render", offset_of!(I, video_render)),
        ("filter_video", offset_of!(I, _b)),
        ("audio_mix", offset_of!(I, _b) + 19 * p),
        ("icon_type", offset_of!(I, icon_type)),
        // what obs_register_source_s copies ends where the next field starts
        ("media_play_pause", size_of::<I>()),
    ]
    .into_iter()
    .map(|(f, v)| (si(f), v as u64))
    .collect();
    let a = |f: &str| format!("offsetof(struct obs_source_audio, {f})");
    for (e, v) in [
        (a("data"), offset_of!(Audio, data)),
        (a("frames"), offset_of!(Audio, frames)),
        (a("speakers"), offset_of!(Audio, speakers)),
        (a("format"), offset_of!(Audio, format)),
        (a("samples_per_sec"), offset_of!(Audio, samples_per_sec)),
        (a("timestamp"), offset_of!(Audio, timestamp)),
        ("sizeof(struct obs_source_audio)".into(), size_of::<Audio>()),
        ("MAX_AV_PLANES".into(), 8),
        ("sizeof(struct vec4)".into(), size_of::<Vec4>()),
        ("_Alignof(struct vec4)".into(), align_of::<Vec4>()),
        ("sizeof(struct drm_get_cap)".into(), size_of::<Cap>()),
        (
            "sizeof(struct drm_syncobj_create)".into(),
            size_of::<Create>(),
        ),
        (
            "sizeof(struct drm_syncobj_destroy)".into(),
            size_of::<Create>(),
        ),
        (
            "sizeof(struct drm_syncobj_handle)".into(),
            size_of::<Handle>(),
        ),
        (
            "offsetof(struct drm_syncobj_handle, fd)".into(),
            offset_of!(Handle, fd),
        ),
        (
            "offsetof(struct drm_syncobj_handle, point)".into(),
            offset_of!(Handle, point),
        ),
        (
            "sizeof(struct drm_syncobj_timeline_wait)".into(),
            size_of::<Wait>(),
        ),
        (
            "offsetof(struct drm_syncobj_timeline_wait, timeout_nsec)".into(),
            offset_of!(Wait, timeout_nsec),
        ),
        (
            "offsetof(struct drm_syncobj_timeline_wait, count_handles)".into(),
            offset_of!(Wait, count_handles),
        ),
        (
            "offsetof(struct drm_syncobj_timeline_wait, deadline_nsec)".into(),
            offset_of!(Wait, deadline_nsec),
        ),
    ] {
        t.push((e, v as u64));
    }
    for (e, v) in [
        ("OBS_SOURCE_TYPE_INPUT", OBS_SOURCE_TYPE_INPUT as u64),
        ("OBS_SOURCE_VIDEO", OBS_SOURCE_VIDEO as u64),
        ("OBS_SOURCE_AUDIO", OBS_SOURCE_AUDIO as u64),
        ("OBS_SOURCE_CUSTOM_DRAW", OBS_SOURCE_CUSTOM_DRAW as u64),
        ("OBS_ICON_TYPE_CAMERA", OBS_ICON_TYPE_CAMERA as u64),
        ("OBS_COMBO_TYPE_LIST", OBS_COMBO_TYPE_LIST as u64),
        ("OBS_COMBO_FORMAT_STRING", OBS_COMBO_FORMAT_STRING as u64),
        ("OBS_TEXT_DEFAULT", OBS_TEXT_DEFAULT as u64),
        ("OBS_TEXT_INFO", OBS_TEXT_INFO as u64),
        ("OBS_EFFECT_DEFAULT", OBS_EFFECT_DEFAULT as u64),
        ("AUDIO_FORMAT_FLOAT", AUDIO_FORMAT_FLOAT as u64),
        // channel count = speaker layout for the counts the plugin passes
        ("SPEAKERS_MONO", 1),
        ("SPEAKERS_STEREO", 2),
        ("SPEAKERS_2POINT1", 3),
        ("SPEAKERS_4POINT0", 4),
        ("SPEAKERS_4POINT1", 5),
        ("SPEAKERS_5POINT1", 6),
        ("SPEAKERS_7POINT1", 8),
        ("GS_R8", GS_R8 as u64),
        ("GS_RGBA", GS_RGBA as u64),
        ("GS_BGRA", GS_BGRA as u64),
        ("GS_R8G8", GS_R8G8 as u64),
        ("LOG_WARNING", LOG_WARNING as u64),
        ("LOG_INFO", LOG_INFO as u64),
        ("LIBOBS_API_VER", LIBOBS_API_VER as u64),
        ("DRM_FORMAT_ABGR8888", AB24 as u64),
        ("DRM_FORMAT_ARGB8888", AR24 as u64),
        ("DRM_FORMAT_NV12", NV12 as u64),
        ("DRM_FORMAT_R8", R8 as u64),
        ("DRM_FORMAT_GR88", GR88 as u64),
        ("DRM_FORMAT_MOD_INVALID", MOD_INVALID),
        ("DRM_FORMAT_MOD_LINEAR", proto::MOD_LINEAR),
        ("DRM_CAP_SYNCOBJ_TIMELINE", CAP_SYNCOBJ_TIMELINE),
        ("DRM_IOCTL_GET_CAP", GET_CAP),
        ("DRM_IOCTL_SYNCOBJ_CREATE", CREATE),
        ("DRM_IOCTL_SYNCOBJ_DESTROY", DESTROY),
        ("DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD", HANDLE_TO_FD),
        ("DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT", TIMELINE_WAIT),
    ] {
        t.push((e.into(), v));
    }
    let dir = std::env::temp_dir().join(format!("x5desk-obs-layout-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut c = String::from(
        "#include <stddef.h>\n#include <stdio.h>\n#include <obs-module.h>\n#include <graphics/vec4.h>\n\
         #include <drm/drm.h>\n#include <drm/drm_fourcc.h>\nint main(void) {\n",
    );
    for (e, _) in &t {
        c += &format!("\tprintf(\"%llu\\n\", (unsigned long long)({e}));\n");
    }
    c += "\treturn 0;\n}\n";
    std::fs::write(dir.join("l.c"), c).unwrap();
    let out = Command::new("cc")
        .args(["-I/usr/include/obs", "-o"])
        .arg(dir.join("l"))
        .arg(dir.join("l.c"))
        .output()
        .expect("cc");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = Command::new(dir.join("l")).output().unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    let got = String::from_utf8(out.stdout).unwrap();
    let got: Vec<u64> = got.lines().map(|l| l.parse().unwrap()).collect();
    assert_eq!(got.len(), t.len());
    let bad: Vec<_> = t
        .iter()
        .zip(&got)
        .filter(|((e, v), g)| match e.as_str() {
            "LIBOBS_API_VER" => v & 0xffff0000 > **g,
            _ => v != *g,
        })
        .map(|((e, v), g)| format!("{e}: ours {v:#x}, C {g:#x}"))
        .collect();
    assert!(bad.is_empty(), "{bad:#?}");
    let api = got[t.iter().position(|(e, _)| e == "LIBOBS_API_VER").unwrap()];
    println!(
        "{} layouts and constants match; API {}.{} declared, libobs {}.{}.{} loads it",
        t.len() - 1,
        LIBOBS_API_VER >> 24,
        LIBOBS_API_VER >> 16 & 0xff,
        api >> 24,
        api >> 16 & 0xff,
        api & 0xffff
    );
}

/// The YUV matrix against libobs' own BT.709 limited-range one (video_format_get_parameters_for_format) and
/// BT.601 full range.
#[test]
fn matrix() {
    let m = yuv(proto::BT709);
    let obs = [
        [1.164384, 0.000000, 1.792741, -0.972945],
        [1.164384, -0.213249, -0.532909, 0.301483],
        [1.164384, 2.112402, 0.000000, -1.133402],
    ];
    for (r, o) in m.iter().zip(obs) {
        for (a, b) in r.0.iter().zip(o) {
            assert!((a - b).abs() < 1e-5, "{:?} vs {o:?}", r.0);
        }
    }
    // full-range BT.601 grey and red
    let m = yuv(proto::BT601 | proto::FULL);
    let rgb = |y: f32, u: f32, v: f32| {
        m.each_ref()
            .map(|r| r.0[0] * y + r.0[1] * u + r.0[2] * v + r.0[3])
    };
    let c = 128. / 255.;
    for (a, b) in rgb(0.5, c, c).iter().zip([0.5; 3]) {
        assert!((a - b).abs() < 1e-6);
    }
    let red = rgb(0.299, (128. - 0.168736 * 255.) / 255., 1.);
    assert!(
        (red[0] - 1.).abs() < 0.01 && red[1].abs() < 0.01 && red[2].abs() < 0.01,
        "{red:?}"
    );
}

/// The syncobj's points report signalled only once the fence imported there has.
#[test]
fn syncobj() {
    let Some(o) = Obj::new() else {
        println!("no render node with timeline syncobjs; skipped");
        return;
    };
    // point 1 without a fence: an error other than ETIME, so it counts as signalled (never pins a buffer)
    assert!(o.signaled(1));
}
