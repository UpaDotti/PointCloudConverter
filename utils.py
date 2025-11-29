import numpy as np


def convert_color(channel_16: np.ndarray) -> np.ndarray:
    # 点群の色をtcp用に変換
    v_255 = (channel_16.astype(np.float32) / 65535.0) * 255.0  # 16bit → 0〜255(float)
    v_linear = gamma_to_linear(v_255)  # ガンマ → リニア
    v_scaled = v_linear * 255.0  # 0〜255 にスケール
    v_clamped = np.clip(v_scaled, 0.0, 255.0)  # 0〜255 にクランプ
    v_u8 = v_clamped.astype(np.uint8)  # uint8 に変換
    return v_u8


def gamma_to_linear(c):
    # sRGB(ガンマ)値をリニアRGBに変換
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def downsample_indices(n, step):
    # 0〜n-1 のインデックスを step 間隔で取得
    return np.arange(0, n, step, dtype=np.int32)


def quantize_u16(value, vmin, vmax):
    """floatをushort16に量子化"""

    # 範囲を取得
    span = float(vmax - vmin)

    # 範囲が0のときは0を返す
    if span <= 0.0:
        return np.zeros_like(value, dtype=np.uint16)

    # 0〜1 に正規化
    normalized = (value - vmin) / span

    # 0〜1 にクリップ
    normalized = np.clip(normalized, 0.0, 1.0)

    # 0〜65535 にスケールして四捨五入
    quantized = np.round(normalized * 65535.0)

    # uint16 にキャスト
    return quantized.astype(np.uint16)
