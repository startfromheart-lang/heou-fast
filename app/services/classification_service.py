"""
图像分类服务
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from fastai.vision.all import *
import pandas as pd
import torch
import numpy as np
from PIL import Image as PILImageNative

from app.core.config import settings
from app.core.gpu_utils import check_gpu_available, initialize_best_gpu

# 配置Hugging Face镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 镜像源
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'        # 超时时间(秒)
os.environ['CURL_CA_BUNDLE'] = ''                    # 避免SSL问题(如有)



class ClassificationService:
    """图像分类服务类"""
    
    def __init__(self):
        self.model = None
        self.device = initialize_best_gpu(silent=True)  # 初始化最佳GPU（独立显卡优先，静默模式）
        self.dls = None
        self.classes = []
        self.is_training = False
        self.segmentation_model = None  # 缓存分割模型
    
    def _load_segmentation_model(self):
        """加载分割模型用于舌头检测"""
        if self.segmentation_model is not None:
            return self.segmentation_model
        
        models_dir = settings.MODELS_DIR
        if models_dir.exists():
            models = list(models_dir.glob("segmentation_*.pkl"))
            if models:
                latest_model = max(models, key=lambda p: p.stat().st_mtime)
                print(f"[舌头检测] 加载分割模型: {latest_model}")
                self.segmentation_model = load_learner(str(latest_model))
                return self.segmentation_model
        
        return None
    
    def detect_and_crop_tongue(
        self,
        image_path: str,
        margin_ratio: float = 0.08
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        检测舌头并裁剪图片
        
        Args:
            image_path: 原始图片路径
            margin_ratio: 裁剪时边缘扩展比例，默认8%
        
        Returns:
            (裁剪后的图片路径, 错误信息) - 成功时错误信息为None，失败时裁剪路径为None
        """
        try:
            print(f"[舌头检测] 开始检测舌头，图片路径: {image_path}")
            
            seg_model = self._load_segmentation_model()
            if seg_model is None:
                return None, "没有可用的分割模型，无法进行舌头检测"
            
            img = PILImage.create(image_path)
            img_array = np.array(img)
            original_h, original_w = img_array.shape[:2]
            
            print(f"[舌头检测] 原图尺寸: {original_w}x{original_h}")
            
            pred_mask, pred_class, outputs = seg_model.predict(img)
            mask_array = np.array(pred_mask)
            
            print(f"[舌头检测] 分割mask形状: {mask_array.shape}")
            print(f"[舌头检测] mask唯一值: {np.unique(mask_array)}")
            
            unique_values = np.unique(mask_array)
            if len(unique_values) <= 1:
                return None, "未检测到舌头，请上传包含舌头的图片"
            
            foreground_class = unique_values[-1] if len(unique_values) > 1 else unique_values[0]
            tongue_mask = (mask_array == foreground_class).astype(np.uint8)
            
            tongue_pixels = np.sum(tongue_mask)
            total_pixels = tongue_mask.size
            tongue_ratio = tongue_pixels / total_pixels
            
            print(f"[舌头检测] 舌头像素占比: {tongue_ratio:.2%}")
            
            if tongue_ratio < 0.001:
                return None, "未检测到舌头，请上传包含舌头的图片"
            
            rows = np.any(tongue_mask, axis=1)
            cols = np.any(tongue_mask, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                return None, "未检测到舌头，请上传包含舌头的图片"
            
            row_indices = np.where(rows)[0]
            col_indices = np.where(cols)[0]
            
            y_min, y_max = row_indices[0], row_indices[-1] + 1
            x_min, x_max = col_indices[0], col_indices[-1] + 1
            
            print(f"[舌头检测] 舌头边界框: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
            
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            
            margin_x = int(bbox_width * margin_ratio)
            margin_y = int(bbox_height * margin_ratio)
            
            crop_x_min = max(0, x_min - margin_x)
            crop_x_max = min(original_w, x_max + margin_x)
            crop_y_min = max(0, y_min - margin_y)
            crop_y_max = min(original_h, y_max + margin_y)
            
            print(f"[舌头检测] 裁剪区域: x=[{crop_x_min}, {crop_x_max}], y=[{crop_y_min}, {crop_y_max}]")
            
            cropped_img = img_array[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cropped_filename = f"tongue_crop_{timestamp}.jpg"
            cropped_path = settings.UPLOAD_DIR / cropped_filename
            settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            
            PILImageNative.fromarray(cropped_img).save(cropped_path, quality=95)
            
            print(f"[舌头检测] 裁剪完成，保存至: {cropped_path}")
            
            return str(cropped_path), None
            
        except Exception as e:
            print(f"[舌头检测] 检测失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, f"舌头检测失败: {str(e)}"
    
    def train(
        self,
        train_path: str,
        valid_path: Optional[str] = None,
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 16,
        model_name: str = "resnet18",
        resume_model: Optional[str] = None,
        pretrained: bool = False,
        model_prefix: str = "classification",
        progress_callback = None
    ) -> Dict[str, Any]:
        """
        训练分类模型

        Args:
            train_path: 训练数据路径
            valid_path: 验证数据路径（可选）
            epochs: 训练轮数
            lr: 学习率
            batch_size: 批次大小
            model_name: 模型名称
            resume_model: 恢复训练的模型路径
            pretrained: 是否使用预训练权重
            model_prefix: 模型保存前缀（如 color, shape, coat）
            progress_callback: 进度回调函数

        Returns:
            训练结果字典
        """
        from datetime import datetime

        print("=" * 60)
        print("[train] 开始训练（文件夹模式）")
        print(f"[train] 训练路径: {train_path}")
        print(f"[train] 验证路径: {valid_path}")
        print(f"[train] 训练轮数: {epochs}")
        print(f"[train] 学习率: {lr}")
        print(f"[train] 批次大小: {batch_size}")
        print(f"[train] 模型名称: {model_name}")
        print(f"[train] 预训练: {pretrained}")
        print(f"[train] 模型前缀: {model_prefix}")
        print("=" * 60)

        self.is_training = True

        # 初始化全局训练进度
        training_progress["classification"] = {
            "is_training": True,
            "status": "loading",
            "message": "正在加载数据集...",
            "epoch": 0,
            "total_epochs": epochs,
            "train_loss": 0.0,
            "valid_loss": 0.0,
            "accuracy": 0.0,
            "start_time": datetime.now().isoformat()
        }
        
        def update_progress(status, message, **kwargs):
            """更新全局训练进度"""
            # 确保所有值都是可 JSON 序列化的
            safe_kwargs = {}
            for key, value in kwargs.items():
                # 转换 numpy 类型为 Python 原生类型
                if value is None:
                    safe_kwargs[key] = None
                elif hasattr(value, 'tolist'):
                    safe_kwargs[key] = value.tolist()
                elif hasattr(value, 'item'):
                    safe_kwargs[key] = float(value.item())
                elif isinstance(value, (np.integer, np.floating)):
                    safe_kwargs[key] = float(value)
                elif isinstance(value, np.ndarray):
                    safe_kwargs[key] = value.tolist()
                elif isinstance(value, (list, tuple)):
                    safe_kwargs[key] = list(value)
                elif isinstance(value, dict):
                    safe_kwargs[key] = value
                elif isinstance(value, (str, int, float, bool)):
                    safe_kwargs[key] = value
                else:
                    # 对于未知类型，尝试转换为字符串
                    try:
                        safe_kwargs[key] = str(value)
                    except Exception as e:
                        safe_kwargs[key] = f"<unserializable: {type(value).__name__}>"

            training_progress["classification"].update({
                "status": status,
                "message": message,
                **safe_kwargs
            })
            if progress_callback:
                progress_callback({"status": status, "message": message, **safe_kwargs})
        
        try:
            update_progress("loading", "正在加载数据集...")

            # 加载数据
            train_path = Path(train_path)
            print(f"✓ 训练路径: {train_path}")
            print(f"✓ 路径存在: {train_path.exists()}")

            if not train_path.exists():
                raise FileNotFoundError(f"训练数据路径不存在: {train_path}")

            print(f"✓ 验证路径: {valid_path}")
            if valid_path:
                valid_path = Path(valid_path)
                print(f"✓ 验证路径存在: {valid_path.exists()}")

            # 使用 ImageDataLoaders.from_folder 创建数据加载器（更稳定的方式）
            print("正在创建数据加载器...")
            try:
                # 尝试最简单的配置
                if valid_path and Path(valid_path).exists():
                    # 如果有独立的验证集路径
                    print("使用独立的训练/验证集模式")
                    dls = ImageDataLoaders.from_folder(
                        train_path.parent,
                        train='train',
                        valid='valid',
                        seed=42,
                        item_tfms=Resize(224, method=ResizeMethod.Squish),
                        bs=batch_size,
                        num_workers=0
                    )
                else:
                    # 使用随机分割
                    print("使用随机分割模式")
                    dls = ImageDataLoaders.from_folder(
                        train_path,
                        valid_pct=0.2,
                        seed=42,
                        item_tfms=Resize(224, method=ResizeMethod.Squish),
                        bs=batch_size,
                        num_workers=0
                    )
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                print(f"创建数据加载器失败详细错误:\n{error_detail}")
                raise Exception(f"创建数据加载器失败: {str(e)}")

            self.dls = dls
            self.classes = dls.vocab

            print(f"✓ 数据加载器创建成功")
            print(f"✓ 训练集样本数: {len(dls.train_ds)}")
            print(f"✓ 验证集样本数: {len(dls.valid_ds)}")
            print(f"✓ 类别数量: {len(dls.vocab)}")
            print(f"✓ 类别: {list(dls.vocab)}")

            update_progress(
                "data_loaded",
                f"数据集加载完成，类别数: {len(self.classes)}",
                classes=list(self.classes),
                train_count=len(dls.train_ds),
                valid_count=len(dls.valid_ds)
            )
            
            # 创建或加载模型
            if resume_model and Path(resume_model).exists():
                update_progress("loading_model", "正在加载已有模型...")
                learn = load_learner(resume_model)
                learn.dls = dls
            else:
                update_progress("creating_model", "正在创建模型...")

                # 如果不使用预训练权重,则从零开始训练
                if pretrained:
                    update_progress("downloading_pretrained", "正在下载预训练模型...")
                    print(f"\n✓ 使用Hugging Face镜像源: {os.environ.get('HF_ENDPOINT', '官方源')}")
                    print("✓ 下载超时时间:", os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT', '60'), "秒")
                    print("✓ 正在下载预训练模型,如果网络较慢可能需要等待...")
                    learn = vision_learner(dls, model_name, metrics=accuracy)
                else:
                    print("使用随机初始化的模型(不从网络下载预训练权重)")
                    # 使用随机初始化的模型
                    learn = vision_learner(dls, model_name, metrics=accuracy, pretrained=False)
            
            # 确保使用GPU
            if check_gpu_available()["available"]:
                # 打印实际使用的GPU信息
                gpu_info = check_gpu_available()
                selected_gpu = gpu_info.get("selected_device", {})
                print(f"\n✓ 图像分类训练使用GPU: {selected_gpu.get('name', 'Unknown')}")
                print(f"✓ 显存: {selected_gpu.get('memory_gb', 'Unknown')} GB")
                print(f"✓ 设备ID: {self.device}")
                print()

                learn.model = learn.model.to(self.device)
                learn.dls.to(self.device)
            
            # 创建自定义训练回调
            class TrainingProgressCallback(Callback):
                def __init__(self, total_epochs):
                    super().__init__()
                    self.total_epochs = total_epochs

                def after_epoch(self):
                    # 更新全局训练进度
                    train_loss = float(self.loss.item()) if hasattr(self.loss, 'item') else float(self.loss)
                    valid_loss = 0.0
                    accuracy = 0.0

                    # 安全地获取验证损失和准确率
                    if hasattr(self.learn, 'recorder') and self.learn.recorder:
                        try:
                            if self.learn.recorder.losses:
                                valid_loss = float(self.learn.recorder.losses[-1])
                            if self.learn.recorder.values and len(self.learn.recorder.values) > 0:
                                # values列表包含每个epoch的损失和指标
                                last_values = self.learn.recorder.values[-1]
                                if len(last_values) > 1:
                                    accuracy = float(last_values[1])
                        except (IndexError, TypeError, ValueError) as e:
                            print(f"Warning: 获取训练指标时出错: {e}")

                    # 确保所有值都是可 JSON 序列化的
                    training_progress["classification"].update({
                        "status": "training",
                        "message": f"正在训练: epoch {self.epoch + 1}/{self.total_epochs}",
                        "epoch": int(self.epoch + 1),
                        "train_loss": float(train_loss),
                        "valid_loss": float(valid_loss),
                        "accuracy": float(accuracy)
                    })
                    print(f"Epoch {self.epoch + 1}/{self.total_epochs} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Accuracy: {accuracy:.4f}")

            update_progress("training", "开始训练...", epochs=epochs)

            # 训练模型（使用自定义进度回调）
            learn.fit_one_cycle(epochs, lr, cbs=[TrainingProgressCallback(epochs)])
            
            # 保存模型
            update_progress("saving", "正在保存模型...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = settings.MODELS_DIR / f"{model_prefix}_{model_name}_{timestamp}.pkl"

            # 确保模型目录存在
            model_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"✓ 模型保存路径: {model_path}")

            learn.export(model_path)
            self.model = learn

            self.is_training = False

            final_metrics = {
                "train_loss": float(learn.recorder.final_record[0]) if hasattr(learn.recorder, 'final_record') else 0.0,
                "valid_loss": float(learn.recorder.final_record[1]) if hasattr(learn.recorder, 'final_record') else 0.0,
                "accuracy": float(learn.recorder.final_record[2]) if hasattr(learn.recorder, 'final_record') else 0.0
            }

            update_progress(
                "completed",
                "训练完成！",
                model_path=str(model_path),
                final_metrics=final_metrics
            )
            
            return {
                "success": True,
                "model_path": str(model_path),
                "classes": list(self.classes),
                "metrics": {
                    "final_accuracy": float(learn.recorder.final_record[2])
                }
            }
            
        except Exception as e:
            self.is_training = False
            update_progress("error", f"训练失败: {str(e)}")
            raise e
    
    def predict(
        self,
        image_path: str,
        model_path: Optional[str] = None,
        detect_tongue: bool = True
    ) -> Dict[str, Any]:
        """
        图像分类预测

        Args:
            image_path: 图像路径
            model_path: 模型路径
            detect_tongue: 是否进行舌头检测和裁剪

        Returns:
            预测结果
        """
        try:
            actual_image_path = image_path
            cropped_image_path = None
            
            if detect_tongue:
                print(f"[分类预测] 开始舌头检测...")
                cropped_path, error_msg = self.detect_and_crop_tongue(image_path)
                
                if error_msg:
                    return {
                        "success": False,
                        "error": error_msg
                    }
                
                if cropped_path:
                    actual_image_path = cropped_path
                    cropped_image_path = cropped_path
                    print(f"[分类预测] 使用裁剪后的图片进行预测: {cropped_path}")
            
            # 处理空字符串的情况
            if model_path is not None and not model_path.strip():
                model_path = None

            # 加载模型
            if model_path:
                learn = load_learner(model_path)
            elif self.model:
                learn = self.model
            else:
                # 尝试自动加载最新的模型
                # 支持新的命名格式：color_*, shape_*, coat_* 和旧的 classification_*
                from pathlib import Path
                models_dir = settings.MODELS_DIR
                if models_dir.exists():
                    models = []
                    for pattern in ["color_*.pkl", "shape_*.pkl", "coat_*.pkl", "classification_*.pkl"]:
                        models.extend(models_dir.glob(pattern))
                    if models:
                        # 按修改时间排序，选择最新的
                        latest_model = max(models, key=lambda p: p.stat().st_mtime)
                        print(f"自动加载最新模型: {latest_model}")
                        learn = load_learner(str(latest_model))
                    else:
                        raise ValueError("没有可用的模型，请先训练或加载模型")
                else:
                    raise ValueError("没有可用的模型，请先训练或加载模型")
            
            # 预测
            img = PILImage.create(actual_image_path)
            pred_class, pred_idx, outputs = learn.predict(img)
            
            # 获取所有类别的概率
            probs = torch.softmax(outputs, dim=0)
            class_probs = []
            for i, cls in enumerate(learn.dls.vocab):
                class_probs.append({
                    "class": cls,
                    "probability": float(probs[i])
                })
            
            result = {
                "success": True,
                "predicted_class": pred_class,
                "confidence": float(probs[pred_idx]),
                "all_probabilities": sorted(class_probs, key=lambda x: x["probability"], reverse=True)
            }
            
            if cropped_image_path:
                result["tongue_detected"] = True
                result["cropped_image_path"] = cropped_image_path
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test(
        self,
        test_path: str,
        model_path: Optional[str] = None,
        progress_callback = None
    ) -> Dict[str, Any]:
        """
        测试模型

        Args:
            test_path: 测试数据路径
            model_path: 模型路径
            progress_callback: 进度回调

        Returns:
            测试结果
        """
        try:
            # 加载模型
            if model_path:
                learn = load_learner(model_path)
            elif self.model:
                learn = self.model
            else:
                raise ValueError("没有可用的模型，请先训练或加载模型")

            # 获取测试图像
            test_path = Path(test_path)
            if not test_path.exists():
                raise FileNotFoundError(f"测试数据路径不存在: {test_path}")

            image_files = get_image_files(test_path)
            total = len(image_files)

            if progress_callback:
                progress_callback({
                    "status": "testing",
                    "message": f"开始测试，共 {total} 张图片",
                    "total": total,
                    "processed": 0
                })

            results = []
            correct = 0

            for i, img_path in enumerate(image_files):
                # 预测
                pred_class, pred_idx, outputs = learn.predict(img_path)
                probs = torch.softmax(outputs, dim=0)

                # 真实类别（从父目录名获取）
                true_class = img_path.parent.name

                # 判断是否正确
                is_correct = (pred_class == true_class)
                if is_correct:
                    correct += 1

                results.append({
                    "image_name": img_path.name,
                    "predicted_class": pred_class,
                    "true_class": true_class,
                    "confidence": float(probs[pred_idx]),
                    "correct": is_correct
                })

                # 进度回调
                if progress_callback and (i + 1) % 10 == 0:
                    progress_callback({
                        "status": "testing",
                        "message": f"已处理 {i + 1}/{total} 张图片",
                        "total": total,
                        "processed": i + 1
                    })

            # 计算准确率
            accuracy = correct / total if total > 0 else 0

            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = settings.RESULTS_DIR / f"classification_test_{timestamp}.csv"
            df = pd.DataFrame(results)
            df.to_csv(result_path, index=False, encoding='utf-8-sig')

            if progress_callback:
                progress_callback({
                    "status": "completed",
                    "message": "测试完成！",
                    "accuracy": accuracy,
                    "result_path": str(result_path)
                })

            return {
                "success": True,
                "accuracy": accuracy,
                "total": total,
                "correct": correct,
                "result_path": str(result_path),
                "results": results[:10]  # 返回前10条结果示例
            }

        except Exception as e:
            if progress_callback:
                progress_callback({
                    "status": "error",
                    "message": f"测试失败: {str(e)}"
                })
            return {
                "success": False,
                "error": str(e)
            }


# 全局训练进度存储
training_progress = {
    "classification": {
        "is_training": False,
        "status": "idle",
        "message": "",
        "epoch": 0,
        "total_epochs": 0,
        "train_loss": 0.0,
        "valid_loss": 0.0,
        "accuracy": 0.0,
        "start_time": "",
        "classes": [],
        "train_count": 0,
        "valid_count": 0
    }
}

# 全局服务实例
classification_service = ClassificationService()
