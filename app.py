# app_pytorch_colorizers.py
"""Streamlit UI that runs both ECCV16 and SIGGRAPH17 PyTorch colorizers locally."""

import io
import os
from typing import Dict, Tuple

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage import color

# -----------------------------------------------------------------------------
# Model definitions adapted from https://github.com/richzhang/colorization
# -----------------------------------------------------------------------------


class BaseColor(nn.Module):
    def __init__(self):
        super().__init__()
        self.l_cent = 50.0
        self.l_norm = 100.0
        self.ab_norm = 110.0

    def normalize_l(self, in_l):
        return (in_l - self.l_cent) / self.l_norm

    def normalize_ab(self, in_ab):
        return in_ab / self.ab_norm

    def unnormalize_ab(self, in_ab):
        return in_ab * self.ab_norm


class ECCVGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super().__init__()

        model1 = [
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(64),
        ]

        model2 = [
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128),
        ]

        model3 = [
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(256),
        ]

        model4 = [
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        model5 = [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        model6 = [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        model7 = [
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        model8 = [
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),
        ]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")

    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))
        return self.unnormalize_ab(self.upsample4(out_reg))


class SIGGRAPHGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d, classes=529):
        super().__init__()

        model1 = [
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(64),
        ]

        model2 = [
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128),
        ]

        model3 = [
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(256),
        ]

        model4 = [
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        model5 = [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        model6 = [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        model7 = [
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        model8up = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)]
        model3short8 = [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)]
        model8 = [
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(256),
        ]

        model9up = [nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)]
        model2short9 = [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)]
        model9 = [
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128),
        ]

        model10up = [nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)]
        model1short10 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)]
        model10 = [
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2),
        ]

        model_class = [nn.Conv2d(256, classes, kernel_size=1, padding=0, dilation=1, stride=1, bias=True)]
        model_out = [
            nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=True),
            nn.Tanh(),
        ]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)
        self.model3short8 = nn.Sequential(*model3short8)
        self.model2short9 = nn.Sequential(*model2short9)
        self.model1short10 = nn.Sequential(*model1short10)

        self.model_class = nn.Sequential(*model_class)
        self.model_out = nn.Sequential(*model_out)

        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_A, input_B=None, mask_B=None):
        if input_B is None:
            input_B = torch.cat((input_A * 0, input_A * 0), dim=1)
        if mask_B is None:
            mask_B = input_A * 0

        conv1_2 = self.model1(
            torch.cat((self.normalize_l(input_A), self.normalize_ab(input_B), mask_B), dim=1)
        )
        conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
        conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
        conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)

        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)
        conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
        conv9_3 = self.model9(conv9_up)
        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        conv10_2 = self.model10(conv10_up)
        out_reg = self.model_out(conv10_2)

        return self.unnormalize_ab(out_reg)


# -----------------------------------------------------------------------------
# Streamlit helpers
# -----------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SPECS = [
    {
        "key": "eccv16",
        "label": "ECCV16 (colorization_release_v2-9b330a0b.pth)",
        "path": "colorization_release_v2-9b330a0b.pth",
        "builder": "eccv16",
        "description": "One-shot colorization model optimized for still images.",
    },
    {
        "key": "siggraph17",
        "label": "SIGGRAPH17 (siggraph17-df00044c.pth)",
        "path": "siggraph17-df00044c.pth",
        "builder": "siggraph17",
        "description": "Interactive model that preserves structure and sharpness.",
    },
]

MODEL_BUILDERS = {
    "eccv16": ECCVGenerator,
    "siggraph17": SIGGRAPHGenerator,
}


def sanitize_state_dict(state_dict):
    state = state_dict
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    if not isinstance(state, dict):
        raise ValueError("Checkpoint format not understood")

    cleaned = {}
    for key, value in state.items():
        new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        cleaned[new_key] = value
    return cleaned


@st.cache_resource(show_spinner=False)
def load_model_cached(model_key: str, model_path: str) -> nn.Module:
    builder_cls = MODEL_BUILDERS[model_key]
    model = builder_cls().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dict = sanitize_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def preprocess_image(image: Image.Image, resize_hw: Tuple[int, int] = (256, 256)) -> Tuple[torch.Tensor, torch.Tensor]:
    img_rgb = image.convert("RGB")
    np_rgb = np.array(img_rgb)
    lab_orig = color.rgb2lab(np_rgb).astype("float32")
    tens_orig_l = torch.from_numpy(lab_orig[:, :, 0]).unsqueeze(0).unsqueeze(0)

    resized = img_rgb.resize(resize_hw, Image.BICUBIC)
    lab_rs = color.rgb2lab(np.array(resized)).astype("float32")
    tens_rs_l = torch.from_numpy(lab_rs[:, :, 0]).unsqueeze(0).unsqueeze(0)
    return tens_orig_l, tens_rs_l


def postprocess(tens_orig_l: torch.Tensor, out_ab: torch.Tensor) -> Image.Image:
    HW_orig = tens_orig_l.shape[2:]
    if out_ab.shape[2:] != HW_orig:
        out_ab = F.interpolate(out_ab, size=HW_orig, mode="bilinear", align_corners=False)
    out_lab = torch.cat([tens_orig_l, out_ab], dim=1)
    lab_np = out_lab[0].permute(1, 2, 0).cpu().numpy()
    rgb = np.clip(color.lab2rgb(lab_np), 0, 1)
    img = Image.fromarray((rgb * 255).astype("uint8"))
    return img


def colorize_with_model(model: nn.Module, image: Image.Image) -> Image.Image:
    tens_orig_l, tens_rs_l = preprocess_image(image)
    tens_rs_l = tens_rs_l.to(DEVICE)
    with torch.inference_mode():
        out_ab = model(tens_rs_l)
    return postprocess(tens_orig_l, out_ab.cpu())


def collect_models() -> Tuple[Dict[str, nn.Module], Dict[str, str]]:
    models = {}
    errors = {}
    for spec in MODEL_SPECS:
        path = spec["path"]
        if not os.path.exists(path):
            errors[spec["label"]] = f"File not found: {path}"
            continue
        try:
            models[spec["key"]] = load_model_cached(spec["key"], path)
        except Exception as exc:  # noqa: BLE001
            errors[spec["label"]] = str(exc)
    return models, errors


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Dual PyTorch Colorizers", layout="wide")

st.title("🎨 ECCV16 + SIGGRAPH17 Colorization")
st.write(
    "This app loads both official PyTorch colorization models locally and runs them on your uploaded grayscale/RGB image."
)

models, load_errors = collect_models()
if load_errors:
    with st.expander("Model load issues", expanded=True):
        for label, err in load_errors.items():
            st.error(f"{label}: {err}")

if not models:
    st.stop()

uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload a grayscale or color photo to start colorizing with both models.")
    st.stop()

input_image = Image.open(uploaded)
st.subheader("Input image")
st.image(input_image, use_column_width=True)

cols = st.columns(len(models))
results = []
for idx, spec in enumerate(MODEL_SPECS):
    key = spec["key"]
    model = models.get(key)
    if model is None:
        continue
    with st.spinner(f"Running {spec['label']}..."):
        output_img = colorize_with_model(model, input_image)
    results.append((spec, output_img))
    with cols[idx]:
        st.markdown(f"### {spec['label']}")
        st.caption(spec["description"])
        st.image(output_img, use_column_width=True)

st.markdown("---")
st.subheader("Download results")
for spec, output_img in results:
    buffer = io.BytesIO()
    output_img.save(buffer, format="PNG")
    st.download_button(
        label=f"Download {spec['key']} result",
        data=buffer.getvalue(),
        file_name=f"colorized_{spec['key']}.png",
        mime="image/png",
        key=f"download-{spec['key']}",
    )
