import numpy as np


def gamma_to_linear(c):
    # sRGB(ガンマ)値をリニアRGBに変換
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def downsample_indices(n, step):
    # 0〜n-1 のインデックスを step 間隔で取得
    return np.arange(0, n, step, dtype=np.int32)


def quantize_u16(value, vmin, vmax):
    # floatをushort16に量子化
    span = vmax - vmin  # 範囲を計算
    normalized = (value - vmin) / span  # 0～1に正規化
    quantized = np.round(normalized * 65535.0)  # 0〜65535にスケールして四捨五入
    return quantized.astype(np.uint16)  # ushortにキャスト
