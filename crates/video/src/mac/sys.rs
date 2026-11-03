//! The few CoreFoundation, CoreMedia, CoreVideo and VideoToolbox calls the macOS backend makes, declared by hand
//! (raw CF pointers, as the C headers have them), and a retained CF reference (`Cf`).
#![allow(non_upper_case_globals, non_snake_case, clippy::upper_case_acronyms)]
use std::ffi::c_void;

pub type OSStatus = i32;
pub type CFTypeRef = *const c_void;
pub type CFIndex = isize;

/// CMTime: `value / timescale` seconds.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CMTime {
    pub value: i64,
    pub timescale: i32,
    pub flags: u32,
    pub epoch: i64,
}

impl CMTime {
    pub const VALID: u32 = 1;

    pub fn new(value: i64, timescale: i32) -> CMTime {
        CMTime {
            value,
            timescale,
            flags: Self::VALID,
            epoch: 0,
        }
    }

    /// kCMTimeInvalid
    pub fn invalid() -> CMTime {
        CMTime::default()
    }
}

#[repr(C)]
pub struct CMSampleTimingInfo {
    pub duration: CMTime,
    pub pts: CMTime,
    pub dts: CMTime,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VTDecompressionOutputCallbackRecord {
    pub callback: VTDecompressionOutputCallback,
    pub refcon: *mut c_void,
}

pub type VTDecompressionOutputCallback = unsafe extern "C" fn(
    refcon: *mut c_void,
    frame_refcon: *mut c_void,
    status: OSStatus,
    flags: u32,
    image: CFTypeRef,
    pts: CMTime,
    duration: CMTime,
);

pub type VTCompressionOutputCallback = unsafe extern "C" fn(
    refcon: *mut c_void,
    frame_refcon: *mut c_void,
    status: OSStatus,
    flags: u32,
    sample: CFTypeRef,
);

// CFNumberType
pub const kCFNumberSInt32Type: CFIndex = 3;
pub const kCFNumberSInt64Type: CFIndex = 4;
pub const kCFNumberFloat64Type: CFIndex = 6;

// CVPixelFormatType: bi-planar 4:2:0, 8-bit (NV12) and 10-bit in the high bits of 16 (P010), video and full range
pub const NV12_VIDEO: u32 = u32::from_be_bytes(*b"420v");
pub const NV12_FULL: u32 = u32::from_be_bytes(*b"420f");
pub const P010_VIDEO: u32 = u32::from_be_bytes(*b"x420");
pub const P010_FULL: u32 = u32::from_be_bytes(*b"xf20");

// CMVideoCodecType
pub const kCMVideoCodecType_H264: u32 = u32::from_be_bytes(*b"avc1");
pub const kCMVideoCodecType_HEVC: u32 = u32::from_be_bytes(*b"hvc1");

// MTLPixelFormat
pub const MTLPixelFormatR8Unorm: usize = 10;
pub const MTLPixelFormatR16Unorm: usize = 20;
pub const MTLPixelFormatRG8Unorm: usize = 30;
pub const MTLPixelFormatRG16Unorm: usize = 60;

pub const kCVAttachmentMode_ShouldPropagate: u32 = 1;
pub const kVTEncodeInfo_FrameDropped: u32 = 1 << 1;

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    pub static kCFAllocatorNull: CFTypeRef;
    pub static kCFBooleanTrue: CFTypeRef;
    pub static kCFBooleanFalse: CFTypeRef;
    pub static kCFTypeDictionaryKeyCallBacks: c_void;
    pub static kCFTypeDictionaryValueCallBacks: c_void;
    pub static kCFTypeArrayCallBacks: c_void;

    pub fn CFRetain(cf: CFTypeRef) -> CFTypeRef;
    pub fn CFRelease(cf: CFTypeRef);
    pub fn CFDictionaryCreateMutable(
        alloc: CFTypeRef,
        capacity: CFIndex,
        keys: *const c_void,
        values: *const c_void,
    ) -> CFTypeRef;
    pub fn CFDictionarySetValue(d: CFTypeRef, key: CFTypeRef, value: CFTypeRef);
    pub fn CFNumberCreate(alloc: CFTypeRef, ty: CFIndex, value: *const c_void) -> CFTypeRef;
    pub fn CFArrayCreate(
        alloc: CFTypeRef,
        values: *const CFTypeRef,
        n: CFIndex,
        callbacks: *const c_void,
    ) -> CFTypeRef;
    pub fn CFArrayGetCount(a: CFTypeRef) -> CFIndex;
    pub fn CFArrayGetValueAtIndex(a: CFTypeRef, i: CFIndex) -> CFTypeRef;
    pub fn CFDictionaryGetValue(d: CFTypeRef, key: CFTypeRef) -> CFTypeRef;
    pub fn CFBooleanGetValue(b: CFTypeRef) -> u8;
}

#[link(name = "CoreMedia", kind = "framework")]
unsafe extern "C" {
    pub static kCMFormatDescriptionExtension_FullRangeVideo: CFTypeRef;
    pub static kCMSampleAttachmentKey_NotSync: CFTypeRef;

    pub fn CMVideoFormatDescriptionCreateFromH264ParameterSets(
        alloc: CFTypeRef,
        n: usize,
        sets: *const *const u8,
        sizes: *const usize,
        nal_len: i32,
        out: *mut CFTypeRef,
    ) -> OSStatus;
    pub fn CMVideoFormatDescriptionCreateFromHEVCParameterSets(
        alloc: CFTypeRef,
        n: usize,
        sets: *const *const u8,
        sizes: *const usize,
        nal_len: i32,
        extensions: CFTypeRef,
        out: *mut CFTypeRef,
    ) -> OSStatus;
    pub fn CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
        desc: CFTypeRef,
        i: usize,
        set: *mut *const u8,
        size: *mut usize,
        count: *mut usize,
        nal_len: *mut i32,
    ) -> OSStatus;
    pub fn CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
        desc: CFTypeRef,
        i: usize,
        set: *mut *const u8,
        size: *mut usize,
        count: *mut usize,
        nal_len: *mut i32,
    ) -> OSStatus;
    pub fn CMFormatDescriptionGetExtension(desc: CFTypeRef, key: CFTypeRef) -> CFTypeRef;
    pub fn CMBlockBufferCreateWithMemoryBlock(
        alloc: CFTypeRef,
        block: *mut c_void,
        len: usize,
        block_alloc: CFTypeRef,
        source: *const c_void,
        offset: usize,
        data_len: usize,
        flags: u32,
        out: *mut CFTypeRef,
    ) -> OSStatus;
    pub fn CMBlockBufferGetDataLength(b: CFTypeRef) -> usize;
    pub fn CMBlockBufferCopyDataBytes(
        b: CFTypeRef,
        offset: usize,
        len: usize,
        dst: *mut c_void,
    ) -> OSStatus;
    pub fn CMSampleBufferCreateReady(
        alloc: CFTypeRef,
        data: CFTypeRef,
        desc: CFTypeRef,
        n: CFIndex,
        n_timing: CFIndex,
        timing: *const CMSampleTimingInfo,
        n_size: CFIndex,
        sizes: *const usize,
        out: *mut CFTypeRef,
    ) -> OSStatus;
    pub fn CMSampleBufferGetDataBuffer(s: CFTypeRef) -> CFTypeRef;
    pub fn CMSampleBufferGetFormatDescription(s: CFTypeRef) -> CFTypeRef;
    pub fn CMSampleBufferGetSampleAttachmentsArray(s: CFTypeRef, create: u8) -> CFTypeRef;
}

#[link(name = "CoreVideo", kind = "framework")]
unsafe extern "C" {
    pub static kCVPixelBufferPixelFormatTypeKey: CFTypeRef;
    pub static kCVPixelBufferIOSurfacePropertiesKey: CFTypeRef;
    pub static kCVPixelBufferMetalCompatibilityKey: CFTypeRef;
    pub static kCVImageBufferYCbCrMatrix_ITU_R_709_2: CFTypeRef;
    pub static kCVImageBufferColorPrimaries_ITU_R_709_2: CFTypeRef;
    pub static kCVImageBufferTransferFunction_ITU_R_709_2: CFTypeRef;
    pub static kCVImageBufferChromaLocationTopFieldKey: CFTypeRef;
    pub static kCVImageBufferChromaLocationBottomFieldKey: CFTypeRef;
    pub static kCVImageBufferChromaLocation_Left: CFTypeRef;
    pub static kCVImageBufferChromaLocation_Center: CFTypeRef;
    pub static kCVImageBufferChromaLocation_TopLeft: CFTypeRef;
    pub static kCVMetalTextureUsage: CFTypeRef;

    pub fn CVPixelBufferGetWidth(pb: CFTypeRef) -> usize;
    pub fn CVPixelBufferGetHeight(pb: CFTypeRef) -> usize;
    pub fn CVPixelBufferCreate(
        alloc: CFTypeRef,
        w: usize,
        h: usize,
        format: u32,
        attrs: CFTypeRef,
        out: *mut CFTypeRef,
    ) -> i32;
    pub fn CVBufferSetAttachment(b: CFTypeRef, key: CFTypeRef, value: CFTypeRef, mode: u32);
    pub fn CVMetalTextureCacheCreate(
        alloc: CFTypeRef,
        attrs: CFTypeRef,
        device: *mut c_void,
        tex_attrs: CFTypeRef,
        out: *mut CFTypeRef,
    ) -> i32;
    pub fn CVMetalTextureCacheCreateTextureFromImage(
        alloc: CFTypeRef,
        cache: CFTypeRef,
        image: CFTypeRef,
        tex_attrs: CFTypeRef,
        format: usize,
        w: usize,
        h: usize,
        plane: usize,
        out: *mut CFTypeRef,
    ) -> i32;
    pub fn CVMetalTextureGetTexture(t: CFTypeRef) -> *mut c_void;
}

#[link(name = "VideoToolbox", kind = "framework")]
unsafe extern "C" {
    pub static kVTCompressionPropertyKey_RealTime: CFTypeRef;
    pub static kVTCompressionPropertyKey_AllowFrameReordering: CFTypeRef;
    pub static kVTCompressionPropertyKey_MaxKeyFrameInterval: CFTypeRef;
    pub static kVTCompressionPropertyKey_ExpectedFrameRate: CFTypeRef;
    pub static kVTCompressionPropertyKey_AverageBitRate: CFTypeRef;
    pub static kVTCompressionPropertyKey_ConstantBitRate: CFTypeRef;
    pub static kVTCompressionPropertyKey_DataRateLimits: CFTypeRef;
    pub static kVTCompressionPropertyKey_ProfileLevel: CFTypeRef;
    pub static kVTCompressionPropertyKey_MinAllowedFrameQP: CFTypeRef;
    pub static kVTCompressionPropertyKey_MaxAllowedFrameQP: CFTypeRef;
    pub static kVTCompressionPropertyKey_ColorPrimaries: CFTypeRef;
    pub static kVTCompressionPropertyKey_TransferFunction: CFTypeRef;
    pub static kVTCompressionPropertyKey_YCbCrMatrix: CFTypeRef;
    pub static kVTCompressionPropertyKey_MaxFrameDelayCount: CFTypeRef;
    pub static kVTProfileLevel_H264_High_AutoLevel: CFTypeRef;
    pub static kVTProfileLevel_HEVC_Main_AutoLevel: CFTypeRef;
    pub static kVTProfileLevel_HEVC_Main10_AutoLevel: CFTypeRef;
    pub static kVTEncodeFrameOptionKey_ForceKeyFrame: CFTypeRef;
    pub static kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder: CFTypeRef;

    pub fn VTDecompressionSessionCreate(
        alloc: CFTypeRef,
        desc: CFTypeRef,
        spec: CFTypeRef,
        attrs: CFTypeRef,
        callback: *const VTDecompressionOutputCallbackRecord,
        out: *mut CFTypeRef,
    ) -> OSStatus;
    pub fn VTDecompressionSessionDecodeFrame(
        s: CFTypeRef,
        sample: CFTypeRef,
        flags: u32,
        frame_refcon: *mut c_void,
        info: *mut u32,
    ) -> OSStatus;
    pub fn VTDecompressionSessionWaitForAsynchronousFrames(s: CFTypeRef) -> OSStatus;
    pub fn VTDecompressionSessionCanAcceptFormatDescription(s: CFTypeRef, desc: CFTypeRef) -> u8;
    pub fn VTDecompressionSessionInvalidate(s: CFTypeRef);

    pub fn VTCompressionSessionCreate(
        alloc: CFTypeRef,
        w: i32,
        h: i32,
        codec: u32,
        spec: CFTypeRef,
        attrs: CFTypeRef,
        data_alloc: CFTypeRef,
        callback: VTCompressionOutputCallback,
        refcon: *mut c_void,
        out: *mut CFTypeRef,
    ) -> OSStatus;
    pub fn VTCompressionSessionPrepareToEncodeFrames(s: CFTypeRef) -> OSStatus;
    pub fn VTCompressionSessionEncodeFrame(
        s: CFTypeRef,
        image: CFTypeRef,
        pts: CMTime,
        duration: CMTime,
        props: CFTypeRef,
        frame_refcon: *mut c_void,
        info: *mut u32,
    ) -> OSStatus;
    pub fn VTCompressionSessionCompleteFrames(s: CFTypeRef, until: CMTime) -> OSStatus;
    pub fn VTCompressionSessionInvalidate(s: CFTypeRef);
    pub fn VTSessionSetProperty(s: CFTypeRef, key: CFTypeRef, value: CFTypeRef) -> OSStatus;
}

/// An owned (+1) CF reference, released on drop.
pub struct Cf(pub CFTypeRef);

// CF objects are thread safe to retain and release; the sessions and buffers here are used by one thread at a time
unsafe impl Send for Cf {}
unsafe impl Sync for Cf {}

impl Cf {
    /// Takes a +1 reference from a Create / Copy call; None for NULL.
    pub fn own(r: CFTypeRef) -> Option<Cf> {
        (!r.is_null()).then_some(Cf(r))
    }

    /// Retains a +0 reference (a Get call or a callback argument).
    pub fn retain(r: CFTypeRef) -> Option<Cf> {
        (!r.is_null()).then(|| Cf(unsafe { CFRetain(r) }))
    }

    pub fn dict() -> Cf {
        Cf(unsafe {
            CFDictionaryCreateMutable(
                std::ptr::null(),
                0,
                &kCFTypeDictionaryKeyCallBacks as *const c_void,
                &kCFTypeDictionaryValueCallBacks as *const c_void,
            )
        })
    }

    pub fn set(&self, key: CFTypeRef, value: CFTypeRef) -> &Cf {
        unsafe { CFDictionarySetValue(self.0, key, value) };
        self
    }

    pub fn i64(v: i64) -> Cf {
        Cf(unsafe { CFNumberCreate(std::ptr::null(), kCFNumberSInt64Type, &v as *const i64 as _) })
    }

    pub fn i32(v: i32) -> Cf {
        Cf(unsafe { CFNumberCreate(std::ptr::null(), kCFNumberSInt32Type, &v as *const i32 as _) })
    }

    pub fn f64(v: f64) -> Cf {
        Cf(unsafe {
            CFNumberCreate(
                std::ptr::null(),
                kCFNumberFloat64Type,
                &v as *const f64 as _,
            )
        })
    }

    pub fn array(v: &[&Cf]) -> Cf {
        let p: Vec<CFTypeRef> = v.iter().map(|c| c.0).collect();
        Cf(unsafe {
            CFArrayCreate(
                std::ptr::null(),
                p.as_ptr(),
                p.len() as CFIndex,
                &kCFTypeArrayCallBacks as *const c_void,
            )
        })
    }
}

impl Drop for Cf {
    fn drop(&mut self) {
        unsafe { CFRelease(self.0) };
    }
}

/// An OSStatus as the crate's error.
pub fn check(what: &'static str, s: OSStatus) -> crate::Result<()> {
    match s {
        0 => Ok(()),
        s => Err(crate::Error::Os(what, s)),
    }
}
