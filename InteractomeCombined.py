import os
import glob
import re
import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.io import imread, imsave
from skimage.morphology import erosion, disk
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List


class ImageAnalyzer:
    """显微镜图像分析工具类"""

    @staticmethod
    def binary_analysis(
            image_path: str,
            mask_path: str,
            erosion_iter: int = 1,
            show_results: bool = False
    ) -> np.ndarray:
        """
        二值化分析图像（仅处理mask指定区域）

        参数:
            image_path: 图像路径
            mask_path: 掩膜路径（白色255为处理区域）
            erosion_iter: 腐蚀迭代次数
            show_results: 是否显示处理过程

        返回:
            处理后的二值图像
        """
        # 读取图像和掩膜
        img = imread(image_path, as_gray=True)
        mask_img = imread(mask_path, as_gray=True)
        # 验证数据
        if img is None or mask_img is None:
            raise FileNotFoundError(f"无法加载图像或掩膜: {image_path} {mask_path}")
        if img.shape != mask_img.shape:
            raise ValueError("图像和掩膜尺寸不匹配")

        # 预处理
        mask = mask_img == 255
        if not np.any(mask):
            raise ValueError("掩膜中没有处理区域")

        img_blur = cv2.medianBlur((img * 255).astype(np.uint8), 3).astype(np.float32) / 255

        # Otsu阈值分割
        thresh = threshold_otsu(img_blur[mask])
        binary_img = np.zeros_like(img_blur, dtype=bool)
        binary_img[mask] = img_blur[mask] > thresh

        # 形态学处理
        selem = disk(1)
        binary_eroded = binary_img.astype(np.uint8)
        for _ in range(erosion_iter):
            binary_eroded = erosion(binary_eroded, selem) & mask.astype(np.uint8)

        # 可视化
        if show_results:
            fig, axes = plt.subplots(1, 4, figsize=(18, 6))
            titles = ['原始图像', '处理掩膜', f'Otsu结果(阈值={thresh:.3f})', f'腐蚀结果(迭代={erosion_iter})']
            for ax, img, title in zip(axes, [img, mask_img, binary_img, binary_eroded], titles):
                ax.imshow(img, cmap='gray')
                ax.set_title(title)
                ax.axis('off')
            plt.tight_layout()
            plt.show()

        return binary_eroded

    @staticmethod
    def parse_filename(filepath: str) -> Tuple[Optional[str], Optional[str]]:
        """
        解析文件名获取标准化层数和通道信息

        参数:
            filepath: 文件路径

        返回:
            (标准化层数, 标准化通道) 或 (None, None)
        """
        filename = os.path.basename(filepath)

        # 匹配层数 (z/t开头) 和通道 (ch/c开头)
        layer_match = re.search(r'[zZtT](\d+)', filename)
        ch_match = re.search(r'(?:ch|CH|c|C|ch_|CH_)[_-]?(\d+)', filename)

        if not layer_match or not ch_match:
            return None, None

        # 完全避免在f-string中使用正则表达式
        channel_num = ''.join([c for c in ch_match.group(1) if c.isdigit()])  # 移除非数字字符
        channel = "ch" + channel_num.zfill(2)  # 使用字符串拼接

        return (
            layer_match.group(1).zfill(3),  # 标准化层数
            channel  # 标准化通道
        )

    @staticmethod
    def load_masks(input_dir: str) -> Dict[str, str]:
        """加载并分类掩膜文件"""
        mask_files = glob.glob(os.path.join(input_dir, 'mask*'))
        if not mask_files:
            raise FileNotFoundError("未找到掩膜文件")

        # 单个掩膜情况
        if len(mask_files) == 1 and os.path.basename(mask_files[0]).startswith('mask.'):
            return {'default': mask_files[0]}

        # 多掩膜情况
        return {
            'ch' + re.sub(r'\D', '', f.split('_')[1].split('.')[0]).zfill(2): f
            for f in mask_files if len(f.split('_')) >= 2
        }

    @classmethod
    def process_series(
            cls,
            input_dir: str,
            output_dir: str,
            erosion_iter: int = 1,
            is_z_stack: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        处理图像序列

        参数:
            input_dir: 输入目录
            output_dir: 输出目录
            erosion_iter: 腐蚀迭代次数
            is_z_stack: 是否为Z-stack数据

        返回:
            包含分析结果的DataFrame
        """
        # 加载图像和掩膜
        img_files = [f for f in glob.glob(os.path.join(input_dir, '*'))
                     if not os.path.basename(f).lower().startswith('mask')]
        masks = cls.load_masks(input_dir)

        # 按层和通道组织数据
        layer_dict = {}
        for img_path in img_files:
            layer, channel = cls.parse_filename(img_path)
            if layer and channel:
                layer_dict.setdefault(layer, {})[channel] = img_path

        if not layer_dict:
            raise ValueError("未找到有效图像数据")

        results = []

        # 处理每层数据
        for layer, channels in layer_dict.items():
            layer_dir = os.path.join(output_dir, f'z{layer}' if is_z_stack else f't{layer}')
            os.makedirs(layer_dir, exist_ok=True)

            binary_images = {}
            for channel, img_path in channels.items():
                mask_path = masks.get('default') or masks.get(channel)
                if not mask_path:
                    print(f"警告: 未找到{channel}通道的掩膜")
                    continue

                try:
                    binary_img = cls.binary_analysis(img_path, mask_path, erosion_iter)
                    imsave(
                        os.path.join(layer_dir, f'binary_{os.path.splitext(os.path.basename(img_path))[0]}.png'),
                        binary_img * 255
                    )
                    binary_images[channel] = binary_img
                except Exception as e:
                    print(f"处理失败 {img_path}: {str(e)}")
                    continue

            # 计算通道间重叠
            channels_sorted = sorted(binary_images.keys())
            for i in range(len(channels_sorted)):
                for j in range(i + 1, len(channels_sorted)):
                    ch1, ch2 = channels_sorted[i], channels_sorted[j]
                    img1, img2 = binary_images[ch1], binary_images[ch2]

                    overlap = np.logical_and(img1, img2)
                    overlap_count = np.sum(overlap)
                    count1, count2 = np.sum(img1 > 0), np.sum(img2 > 0)

                    results.append({
                        'Z层' if is_z_stack else '时间点': layer,
                        '通道1': ch1,
                        '通道2': ch2,
                        '重叠像素数': overlap_count,
                        '通道1阳性数': count1,
                        '通道2阳性数': count2,
                        '重叠比(通道1)': overlap_count / max(count1, 1),
                        '重叠比(通道2)': overlap_count / max(count2, 1)
                    })

        return pd.DataFrame(results) if results else None

    @classmethod
    def process_z_stack_series(
            cls,
            input_dir: str,
            output_dir: str,
            erosion_iter: int = 1
    ) -> Optional[pd.DataFrame]:
        """
        处理Z-stack图像序列并汇总各层重叠数据

        参数:
            input_dir: 输入目录路径（包含Z-stack图像）
            output_dir: 输出目录路径
            erosion_iter: 腐蚀迭代次数

        返回:
            汇总统计DataFrame
        """
        # 处理Z-stack序列（得到各层的DataFrame）
        df = cls.process_series(input_dir, output_dir, erosion_iter, is_z_stack=True)
        if df is None:
            print("未生成任何结果数据")
            return None

        # 保存原始数据
        raw_path = os.path.join(output_dir, 'z_stack_各层统计.csv')
        df.to_csv(raw_path, index=False, encoding='utf_8_sig')

        # 生成汇总统计
        summary = df.groupby(['通道1', '通道2']).agg({
            '重叠像素数': 'sum',
            '通道1阳性数': 'sum',
            '通道2阳性数': 'sum',
            'Z层': 'count'
        }).reset_index()

        summary['重叠比(通道1)'] = summary['重叠像素数'] / summary['通道1阳性数'].clip(lower=1)
        summary['重叠比(通道2)'] = summary['重叠像素数'] / summary['通道2阳性数'].clip(lower=1)

        summary = summary[[
            '通道1', '通道2', '重叠像素数',
            '通道1阳性数', '通道2阳性数',
            '重叠比(通道1)', '重叠比(通道2)',
            'Z层'
        ]].rename(columns={'Z层': 'Z层数'})

        summary_path = os.path.join(output_dir, 'z_stack_汇总统计.csv')
        summary.to_csv(summary_path, index=False, encoding='utf_8_sig')

        return summary

if __name__ == "__main__":
    # 使用示例
    analyzer = ImageAnalyzer()
    '''
    # 处理时间序列
    analyzer.process_series(
        input_dir="path/to/time_series",
        output_dir="path/to/output",
        erosion_iter=1
    )
    '''
    # 处理Z-stack
    analyzer.process_z_stack_series(
        input_dir=r"C:\Users\谈自主\Desktop\OrganelleInteractome\20250327",
        output_dir=r"C:\Users\谈自主\Desktop\OrganelleInteractome\20250327\output",
        erosion_iter=1
    )