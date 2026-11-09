<div align="center">
  <img src="assets/filmclear.png" width="128" height="128" alt="FilmClear 图标">
  <h1>FilmClear</h1>
  <p>在本地检查和修复胶片扫描照片的灰尘、毛絮、划痕与片面破损。</p>
  <p><a href="https://github.com/JackerKun/FilmClear/releases/latest">下载最新版本</a> · <a href="#快速上手">快速上手</a> · <a href="#从源码运行与打包">源码与打包</a></p>
</div>

## 下载与运行

在 [GitHub Releases](https://github.com/JackerKun/FilmClear/releases/latest) 选择与系统匹配的归档。发布页附有 `SHA256SUMS.txt`，可用于校验下载文件。

| 系统 | 下载文件 | 解压后启动 |
| --- | --- | --- |
| macOS Apple Silicon | `FilmClear-macOS-arm64.zip` | 打开 `FilmClear.app` |
| macOS Intel | `FilmClear-macOS-x64.zip` | 打开 `FilmClear.app` |
| Windows x64 | `FilmClear-Windows-x64.zip` | 进入 `FilmClear` 文件夹，打开 `FilmClear.exe` |
| Linux x64 | `FilmClear-Linux-x64.tar.gz` | 在解压目录运行 `./FilmClear/FilmClear` |

Windows 和 Linux 请保留整个解压目录。macOS 构建使用临时签名，跨机器下载时系统可能要求确认。Linux 需要图形桌面及其 Qt 系统库。

## 快速上手

1. **导入照片**。打开照片或文件夹，也可将文件拖入窗口。支持 JPEG、PNG、TIFF、WebP、BMP。
2. **扫描缺陷**。默认「主体优先」关注人脸和平坦背景，对山林纹理和齿孔更保守。「快速预览」用缩放后的图像查找候选，较细的划痕和毛絮会以更高分辨率复查。可切换到「原图精查」或「全画面」。
3. **核对候选**。候选默认勾选并显示标记。悬停列表可查看局部，点击可在原图定位。拖框「补查」或在瑕疵附近「描画」，程序会在原图像素上寻找实际缺陷轮廓。
4. **修复与对比**。可框选勾选、忽略或修复，按住 Shift 可叠加多个勾选框。修复和忽略可撤销。前后对比支持滑动分割、左右并排、上下排列，双画面同步缩放和平移。
5. **导出**。按原始像素尺寸导出，可跟随原格式或统一为 JPEG、PNG、TIFF。PNG 和 TIFF 支持 16 位输出。批量导出可选择照片、查看进度和取消剩余任务。

切换照片会保留当前会话中的候选、勾选和编辑历史。照片在本机处理。

## 红外扫描

- VueScan 导出时可选择 **Raw file**、**64 bit RGBI**、**Raw output with = Scan**。FilmClear 会识别 TIFF 的内嵌红外通道。
- 同目录的 `照片名_IR.tif` 或 `照片名_infrared.tiff` 可自动关联。也可从文件菜单指定已对齐的红外图，或手动将 TIFF 第四通道作为红外。
- 红外信号不足时可切到「仅 RGB」。导出时选择「复制原文件」可保留内嵌红外通道及原文件元数据。

[真实 VueScan 测试样本的来源与裁切说明](demo/IR-SOURCE.md)

## 快捷键

| 操作 | 按键 |
| --- | --- |
| 临时抓手平移 | 按住空格拖动，松开恢复当前工具 |
| 前后对比 | `B` |
| 显示或隐藏标记 | `M` |
| 适应窗口 / 原始像素 | `F` / `1` |
| 打开 / 导出 / 批量导出 | `⌘O` / `⌘S` / `⇧⌘S`（macOS） |
| 撤销 / 重做 | `⌘Z` / `⇧⌘Z`（macOS） |

## 从源码运行与打包

需要 Python 3.10 或更新版本。macOS 和 Linux 可在项目根目录运行：

```bash
./start.sh
./start.sh --test -v
```

Windows 可在命令提示符中运行：

```bat
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

个人扫描样本放在 `demo/`。干净的仓库检出会运行合成样本测试，并跳过依赖本地扫描照片的用例。

| 本地打包 | 命令 | 输出 |
| --- | --- | --- |
| macOS | `./build.sh` | `dist/macos/FilmClear.app` |
| Windows | `build.bat` | `dist/windows/FilmClear/FilmClear.exe` |
| Linux | `./build.sh` | `dist/linux/FilmClear/FilmClear` |

打包脚本使用独立的 `.venv-build`，并将运行依赖、图标和 YuNet 权重放入产物。首次建立打包环境时，可通过 `FILMCLEAR_PYTHON` 指定 Python 可执行文件。GitHub Actions 在推送 `v*` 标签后构建各平台归档、检查程序启动，并发布下载和校验文件。

## 模型与素材

「主体优先」使用 YuNet 定位人脸，缺陷轮廓由原图检测生成。模型权重来自 [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)，随附 [MIT 许可证](models/YuNet-LICENSE.txt)。应用图标的矢量源文件位于 [assets/filmclear.svg](assets/filmclear.svg)。
