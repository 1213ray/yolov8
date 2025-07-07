
"""
# ==================================================================================
# YOLOv8 半监督学习集成管线 - 总协调器
# ==================================================================================
#
# 作者: Gemini AI
# 日期: 2025-07-05
#
# 说明:
#   这是整个半监督学习项目的总入口和协调器。
#   它按照预定义的四个阶段，依次调用各个模块，完成从数据准备、模型训练、
#   伪标签生成、伪标签提纯到最终模型再训练的全过程。
#
#   运行此脚本前，请确保：
#   1. `src/config.py` 中的配置已根据您的需求调整完毕。
#   2. 初始的带标签数据已放入 `data/labelled/` 目录。
#   3. 未标注数据已放入 `data/unlabelled/` 目录。
#
# 执行顺序:
#   - Stage 1a: K-Fold Trainer -> 为 cGAN 生成无偏训练数据。
#   - Stage 1b: Initial Training -> 训练一个初始的“教师模型”。
#   - Stage 2:  Pseudo-Labeling -> 使用教师模型生成原始伪标签。
#   - Stage 3:  cGAN Training & Refinement -> 训练 cGAN 校准器并用它提纯伪标签。
#   - Stage 4:  Final Training -> 使用“带标签数据 + 提纯后的伪标签”训练最终模型。
#
# ==================================================================================
"""

import yaml
import shutil

# 导入所有需要的模块和配置
from src.config import *
from src.train_yolo import train_yolo_model, create_dataset_yaml
from src.k_fold_trainer import run_kfold_training_and_inference
from src.stage2_pseudo_labeler import generate_pseudo_labels
from src.cgan_calibrator import run_cgan_training
from src.stage3_refine_pseudo import refine_all_pseudo_labels

def main():
    """执行完整管线"""
    print("======================================================")
    print("===   YOLOv8 半监督学习集成管线启动   ===")
    print("======================================================")

    # --- STAGE 1a: K-Fold 交叉验证，为 cGAN 准备数据 ---
    # -----------------------------------------------------
    print_stage_header("1a", "K-Fold 交叉验证与推理")
    run_kfold_training_and_inference()

    # --- STAGE 1b: 训练初始教师模型 ---
    # -----------------------------------
    print_stage_header("1b", "训练初始教师模型")
    teacher_yaml_path = STAGE1_TEACHER_MODEL_DIR / "teacher_dataset.yaml"
    create_dataset_yaml(
        yaml_path=teacher_yaml_path,
        train_dir=LABELLED_DATA_DIR, # 使用全部带标签数据进行训练
        val_dir=LABELLED_DATA_DIR    # 验证集也用它，或者您可以单独划分一个
    )
    teacher_model_path = train_yolo_model(
        data_yaml_path=teacher_yaml_path,
        epochs=TEACHER_TRAIN_EPOCHS,
        batch_size=TEACHER_TRAIN_BATCH,
        img_size=TEACHER_TRAIN_IMG_SIZE,
        project_name="Stage1b_Teacher_Model_Training",
        model_name=TEACHER_MODEL_NAME,
        base_model=YOLO_BASE_MODEL
    )
    print(f"教师模型训练完成，已保存至: {teacher_model_path}")

    # --- STAGE 2: 生成两阶段伪标签 ---
    # -----------------------------------
    print_stage_header("2", "生成两阶段伪标签")
    generate_pseudo_labels()

    # --- STAGE 3: 训练 cGAN 并提纯伪标签 ---
    # -----------------------------------------
    print_stage_header("3", "训练 cGAN 校准器并提纯伪标签")
    # Part 1: 训练 cGAN
    run_cgan_training()
    # Part 2: 使用 cGAN 提纯
    refine_all_pseudo_labels()

    # --- STAGE 4: 训练最终学生模型 ---
    # -----------------------------------
    print_stage_header("4", "训练最终学生模型")
    # a. 准备最终的混合数据集 (带标签数据 + 提纯后的伪标签数据)
    final_dataset_dir = OUTPUTS_DIR / "final_training_dataset"
    final_train_img = final_dataset_dir / "train" / "images"; final_train_img.mkdir(parents=True, exist_ok=True)
    final_train_lab = final_dataset_dir / "train" / "labels"; final_train_lab.mkdir(parents=True, exist_ok=True)
    final_val_img = final_dataset_dir / "val" / "images"; final_val_img.mkdir(parents=True, exist_ok=True)
    final_val_lab = final_dataset_dir / "val" / "labels"; final_val_lab.mkdir(parents=True, exist_ok=True)

    print("正在准备最终的混合训练数据...")
    # 复制所有带标签数据到 final_train
    copy_files(LABELLED_DATA_DIR / "images", final_train_img)
    copy_files(LABELLED_DATA_DIR / "labels", final_train_lab)
    # 复制所有提纯后的伪标签数据到 final_train
    copy_files(STAGE4_PSEUDO_REFINED_DIR / "images", final_train_img)
    copy_files(STAGE4_PSEUDO_REFINED_DIR / "labels", final_train_lab)
    
    # 使用原始的带标签数据作为最终的验证集
    copy_files(LABELLED_DATA_DIR / "images", final_val_img)
    copy_files(LABELLED_DATA_DIR / "labels", final_val_lab)
    print("最终混合数据集准备完成！")

    # b. 创建最终的 .yaml 文件
    final_yaml_path = final_dataset_dir / "final_dataset.yaml"
    create_dataset_yaml(final_yaml_path, final_dataset_dir / "train", final_dataset_dir / "val")

    # c. 训练最终模型
    final_model_path = train_yolo_model(
        data_yaml_path=final_yaml_path,
        epochs=FINAL_TRAIN_EPOCHS,
        batch_size=FINAL_TRAIN_BATCH,
        img_size=FINAL_TRAIN_IMG_SIZE,
        project_name="Stage4_Final_Model_Training",
        model_name=FINAL_MODEL_NAME,
        base_model=YOLO_BASE_MODEL # 从头开始训练，或者也可以用 teacher_model 作为起点
    )

    print("\n======================================================")
    print("===       所有阶段执行完毕！管线成功结束。       ===")
    print(f"=== 最终模型已保存至: {final_model_path} ===")
    print("======================================================")

def print_stage_header(stage_num, stage_name):
    print(f"\n{'='*25} STAGE {stage_num}: {stage_name} {'='*25}")

def copy_files(source_dir, dest_dir):
    for f in source_dir.glob("*"):
        shutil.copy(f, dest_dir / f.name)

if __name__ == '__main__':
    main()
