# HDR 导出样片

[HDR-window-16bit.tif](HDR-window-16bit.tif) 是一张合成的 16 位 sRGB 正片扫描，带有扫描黑边、灰点、弯曲毛絮和细划痕。窗外云层和灯光用于比较高光，暗部和窗框可用于核对灰度、色彩和修复边缘。

1. 在项目根目录运行 `./start.sh`，用「添加照片」导入这张 TIFF。
2. 先修复或裁切，再打开「导出」。用「仅导出已应用的编辑」分别导出普通 JPEG、Ultra HDR JPEG 和 10 位 PQ HEIC，三个版本应具有相同取景。
3. 用支持 HDR 的照片应用和显示器打开文件，重点比较窗外云层和台灯。若 Mac 上的查看器将 Ultra HDR JPEG 显示为 SDR，可以先用 PQ HEIC 对比。HTML 对照页使用 SDR 预览。

下列参考文件使用同一张原始 TIFF 导出，保留黑边和模拟缺陷，方便在改动前直接比对。

- [普通 JPEG](HDR-window-reference/HDR-window-SDR.jpg)
- [Ultra HDR JPEG](HDR-window-reference/HDR-window-UltraHDR.jpg)
- [10 位 PQ HEIC](HDR-window-reference/HDR-window-PQ.heic)

也可以直接把 Ultra HDR JPEG 或 PQ HEIC 导入 FilmClear，裁切或修复后选择「跟随原文件格式」保留 HDR，或切换到另一种 HDR 格式。普通 JPEG 可用于 SDR 对照。

HDR 曲线最多将高光提亮约一档，原扫描中已剪切的高光仍会缺少细节。
