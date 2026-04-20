use std::borrow::Cow;
use std::convert::Into;
use std::iter::Iterator;
use std::path::Path;
use std::sync::LazyLock;

use strum::{Display, EnumIter, IntoEnumIterator, IntoStaticStr};
use tokio::process::Command;

use crate::error::FFPipelineError;

static KNOWN_ACCELS: LazyLock<Vec<&'static str>> =
    LazyLock::new(|| KnownHardwareAccel::iter().map(|x| x.into()).collect());

static KNOWN_FILTERS: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    KnownVideoFilter::iter()
        .map(|x| x.into())
        .collect::<Vec<&str>>()
});

#[derive(Display, EnumIter, IntoStaticStr, Debug, PartialEq)]
pub enum KnownHardwareAccel {
    #[strum(serialize = "cuda")]
    Cuda,
    #[strum(serialize = "qsv")]
    Qsv,
    #[strum(serialize = "vaapi")]
    Vaapi,
    #[strum(serialize = "videotoolbox")]
    VideoToolbox,
    #[strum(serialize = "vulkan")]
    Vulkan,
    #[strum(serialize = "opencl")]
    Opencl,
}

/// Convert a KnownHardwareAccel to a Cow-wrapped &'static str.
impl From<KnownHardwareAccel> for Cow<'static, str> {
    fn from(value: KnownHardwareAccel) -> Self {
        Cow::<'static, str>::from(<KnownHardwareAccel as Into<&'static str>>::into(value))
    }
}

#[derive(Display, EnumIter, IntoStaticStr, Debug, PartialEq)]
pub enum KnownVideoFilter {
    #[strum(serialize = "bwdif")]
    Bwdif,
    #[strum(serialize = "libplacebo")]
    LibPlacebo,
    #[strum(serialize = "pad_cuda")]
    PadCuda,
    #[strum(serialize = "pad_vaapi")]
    PadVaapi,
    #[strum(serialize = "scale_cuda")]
    ScaleCuda,
    #[strum(serialize = "scale_vaapi")]
    ScaleVaapi,
    #[strum(serialize = "scale_vulkan")]
    ScaleVulkan,
    #[strum(serialize = "tonemap_opencl")]
    TonemapOpencl,
    #[strum(serialize = "vpp_qsv")]
    VppQsv,
    #[strum(serialize = "w3fdif")]
    W3fdif,
    #[strum(serialize = "yadif")]
    Yadif,
}

#[derive(Debug, Clone, Default)]
pub struct FfmpegInfo {
    hwaccels: Vec<String>,
    video_filters: Vec<String>,
    pub(crate) preferred_filters: Vec<String>,
}

impl FfmpegInfo {
    pub async fn load(
        path: &Path,
        disabled_filters: &[String],
        preferred_filters: &[String],
    ) -> Result<FfmpegInfo, FFPipelineError> {
        let hwaccels = Self::load_hw_accels(path).await?;
        let video_filters = Self::load_video_filters(path, disabled_filters).await?;

        // filter preferred by known video filters
        let mut preferred: Vec<String> = Vec::new();
        for filter in preferred_filters {
            if video_filters.contains(filter) {
                preferred.push(filter.clone());
            }
        }

        Ok(FfmpegInfo {
            hwaccels,
            video_filters,
            preferred_filters: preferred,
        })
    }

    pub fn has_hw_accel(&self, hw_accel: &KnownHardwareAccel) -> bool {
        let accel_string = hw_accel.to_string();
        self.hwaccels.iter().any(|f| f == &accel_string)
    }

    pub fn has_video_filter(&self, filter: &KnownVideoFilter) -> bool {
        let filter_string = filter.to_string();
        self.video_filters.iter().any(|f| f == &filter_string)
    }

    async fn load_hw_accels(path: &Path) -> Result<Vec<String>, FFPipelineError> {
        let output = Command::new(path)
            .args(["-hide_banner", "-hwaccels"])
            .output()
            .await
            .map_err(|_| FFPipelineError::FfmpegCapabilitiesError(String::from("hwaccels")))?;

        let stdout = String::from_utf8_lossy(&output.stdout);

        let mut accels: Vec<String> = Vec::new();

        for line in stdout.lines() {
            let trimmed = line.trim();

            if trimmed.contains(":") || trimmed.is_empty() {
                continue;
            }

            if KNOWN_ACCELS.contains(&trimmed) {
                accels.push(trimmed.to_owned());
            }
        }

        Ok(accels)
    }

    async fn load_video_filters(
        path: &Path,
        disabled_filters: &[String],
    ) -> Result<Vec<String>, FFPipelineError> {
        let output = Command::new(path)
            .args(["-hide_banner", "-filters"])
            .output()
            .await
            .map_err(|_| FFPipelineError::FfmpegCapabilitiesError(String::from("filters")))?;

        let stdout = String::from_utf8_lossy(&output.stdout);

        let mut filters: Vec<String> = Vec::new();

        for line in stdout.lines() {
            //  .. scale_cuda        V->V       GPU accelerated video resizer
            if let Some(filter) = line.split_whitespace().nth(1)
                && KNOWN_FILTERS.contains(&filter)
                && !disabled_filters.iter().any(|f| f == filter)
            {
                filters.push(filter.to_owned());
            }
        }

        Ok(filters)
    }
}
