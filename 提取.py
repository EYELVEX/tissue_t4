import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_nifti(file_path):
    """加载NIfTI文件并返回数据和头文件信息"""
    nii = nib.load(file_path)
    data = nii.get_fdata()
    header = nii.header
    return data, header

def get_t4_z_range(t4_mask):
    """获取T4胸椎在z轴方向的范围"""
    # 找到T4标签(值为40)的所有位置
    t4_coords = np.where(t4_mask == 40)
    
    if len(t4_coords[2]) == 0:
        print("警告: 未找到T4胸椎标签(值=40)")
        return None, None
    
    # 获取T4在z轴方向的最小和最大索引
    z_min = np.min(t4_coords[2])
    z_max = np.max(t4_coords[2])
    
    return z_min, z_max

def calculate_tissue_volumes(tissue_mask, t4_z_min, t4_z_max, voxel_volume):
    """计算T4范围内各组织的体积"""
    volumes = {}
    tissue_labels = {1: '皮下脂肪', 2: '纵隔脂肪', 3: '骨骼肌'}
    
    # 提取T4高度范围内的组织数据
    if t4_z_min is not None and t4_z_max is not None:
        tissue_roi = tissue_mask[:, :, t4_z_min:t4_z_max+1]
    else:
        print("警告: T4范围无效，使用整个volume")
        tissue_roi = tissue_mask
    
    for label_value, label_name in tissue_labels.items():
        # 计算该标签的体素数量
        voxel_count = np.sum(tissue_roi == label_value)
        # 计算体积 (单位: mm³)
        volume_mm3 = voxel_count * voxel_volume
        # 转换为cm³
        volume_cm3 = volume_mm3 / 1000
        
        volumes[label_name] = {
            'voxel_count': int(voxel_count),
            'volume_mm3': float(volume_mm3),
            'volume_cm3': float(volume_cm3)
        }
    
    return volumes, tissue_roi

def create_cropped_mask(original_mask, t4_z_min, t4_z_max):
    """创建裁剪后的掩膜（仅包含T4高度范围）"""
    if t4_z_min is None or t4_z_max is None:
        print("警告: T4范围无效，返回原始掩膜")
        return original_mask
    
    # 创建新的掩膜，只保留T4高度范围的切片
    cropped_mask = original_mask[:, :, t4_z_min:t4_z_max+1]
    return cropped_mask

def save_cropped_mask(cropped_mask, original_header, output_path, t4_z_min, t4_z_max):
    """保存裁剪后的掩膜文件"""
    # 复制原始header
    new_header = original_header.copy()
    
    # 更新header中的维度信息
    new_shape = cropped_mask.shape
    new_header.set_data_shape(new_shape)
    
    # 更新仿射矩阵以反映z轴的偏移
    affine = new_header.get_best_affine()
    if affine is not None:
        # 调整z轴的偏移量
        z_spacing = affine[2, 2]  # z轴体素间距
        affine[2, 3] += t4_z_min * z_spacing  # 调整z轴原点
    
    # 创建新的NIfTI图像
    new_nii = nib.Nifti1Image(cropped_mask.astype(np.int16), affine, new_header)
    
    # 保存文件
    nib.save(new_nii, output_path)
    print(f"  裁剪后掩膜已保存: {output_path}")

def process_all_patients():
    """处理所有病人的数据"""
    # 定义文件夹路径
    images_dir = Path('imagesTr')
    t4_labels_dir = Path('labelsTr_T4')
    tissue_labels_dir = Path('labelsTr_tissue')
    
    # 创建输出文件夹用于保存裁剪后的掩膜
    cropped_tissue_dir = Path('labelsTr_tissue_cropped')
    cropped_t4_dir = Path('labelsTr_T4_cropped')
    cropped_images_dir = Path('imagesTr_cropped')
    
    # 创建输出目录
    for output_dir in [cropped_tissue_dir, cropped_t4_dir, cropped_images_dir]:
        output_dir.mkdir(exist_ok=True)
        print(f"创建输出目录: {output_dir}")
    
    # 检查输入文件夹是否存在
    if not all([images_dir.exists(), t4_labels_dir.exists(), tissue_labels_dir.exists()]):
        print("错误: 请确保以下文件夹存在:")
        print(f"- {images_dir}")
        print(f"- {t4_labels_dir}")
        print(f"- {tissue_labels_dir}")
        return None
    
    results = []
    
    # 获取所有病人的文件列表
    t4_files = sorted(list(t4_labels_dir.glob('*.nii.gz')))
    
    for t4_file in t4_files:
        patient_id = t4_file.stem.replace('.nii', '')
        print(f"正在处理病人: {patient_id}")
        
        # 构建对应的文件路径
        image_file = images_dir / f"{patient_id}.nii.gz"
        tissue_file = tissue_labels_dir / f"{patient_id}.nii.gz"
        
        # 检查文件是否存在
        if not all([image_file.exists(), tissue_file.exists()]):
            print(f"警告: 病人 {patient_id} 的某些文件不存在，跳过")
            continue
        
        try:
            # 加载原始图像文件
            image_data, image_header = load_nifti(image_file)
            
            # 加载T4标签文件
            t4_data, t4_header = load_nifti(t4_file)
            
            # 加载组织标签文件
            tissue_data, tissue_header = load_nifti(tissue_file)
            
            # 计算体素体积 (mm³)
            voxel_dims = tissue_header.get_zooms()[:3]  # x, y, z方向的体素尺寸
            voxel_volume = np.prod(voxel_dims)
            
            # 获取T4胸椎的z轴范围
            t4_z_min, t4_z_max = get_t4_z_range(t4_data)
            
            if t4_z_min is None:
                print(f"警告: 病人 {patient_id} 未找到T4标签，跳过")
                continue
            
            # 计算各组织体积
            volumes, tissue_roi = calculate_tissue_volumes(tissue_data, t4_z_min, t4_z_max, voxel_volume)
            
            # 创建裁剪后的掩膜
            cropped_tissue_mask = create_cropped_mask(tissue_data, t4_z_min, t4_z_max)
            cropped_t4_mask = create_cropped_mask(t4_data, t4_z_min, t4_z_max)
            cropped_image = create_cropped_mask(image_data, t4_z_min, t4_z_max)
            
            # 保存裁剪后的文件
            cropped_tissue_path = cropped_tissue_dir / f"{patient_id}.nii.gz"
            cropped_t4_path = cropped_t4_dir / f"{patient_id}.nii.gz"
            cropped_image_path = cropped_images_dir / f"{patient_id}.nii.gz"
            
            save_cropped_mask(cropped_tissue_mask, tissue_header, cropped_tissue_path, t4_z_min, t4_z_max)
            save_cropped_mask(cropped_t4_mask, t4_header, cropped_t4_path, t4_z_min, t4_z_max)
            save_cropped_mask(cropped_image, image_header, cropped_image_path, t4_z_min, t4_z_max)
            
            # 保存结果
            patient_result = {
                'patient_id': patient_id,
                't4_z_min': int(t4_z_min),
                't4_z_max': int(t4_z_max),
                't4_height_slices': int(t4_z_max - t4_z_min + 1),
                'voxel_volume_mm3': float(voxel_volume),
                'cropped_shape': f"{cropped_tissue_mask.shape[0]}x{cropped_tissue_mask.shape[1]}x{cropped_tissue_mask.shape[2]}"
            }
            
            # 添加各组织的体积信息
            for tissue_name, volume_info in volumes.items():
                patient_result[f'{tissue_name}_voxel_count'] = volume_info['voxel_count']
                patient_result[f'{tissue_name}_volume_mm3'] = volume_info['volume_mm3']
                patient_result[f'{tissue_name}_volume_cm3'] = volume_info['volume_cm3']
            
            results.append(patient_result)
            
            print(f"  T4范围: slice {t4_z_min} - {t4_z_max} (共{t4_z_max-t4_z_min+1}层)")
            print(f"  裁剪后尺寸: {cropped_tissue_mask.shape}")
            for tissue_name, volume_info in volumes.items():
                print(f"  {tissue_name}: {volume_info['volume_cm3']:.2f} cm³")
            
        except Exception as e:
            print(f"处理病人 {patient_id} 时出错: {str(e)}")
            continue
    
    print(f"\n裁剪后的文件已保存到以下目录:")
    print(f"- 组织掩膜: {cropped_tissue_dir}")
    print(f"- T4掩膜: {cropped_t4_dir}")
    print(f"- 原始图像: {cropped_images_dir}")
    
    return results

def save_results_to_csv(results, filename='t4_tissue_volumes.csv'):
    """保存结果到CSV文件"""
    if not results:
        print("没有结果可保存")
        return None
    
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"结果已保存到: {filename}")
    return df

def create_visualization(df, save_plots=True):
    """创建可视化图表"""
    if df is None or df.empty:
        print("没有数据可视化")
        return
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 准备数据用于可视化
    tissue_columns = [col for col in df.columns if col.endswith('_volume_cm3')]
    tissue_names = [col.replace('_volume_cm3', '') for col in tissue_columns]
    
    # 1. 各组织体积分布箱线图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('T4胸椎高度范围内组织体积分析', fontsize=16, fontweight='bold')
    
    # 箱线图
    volume_data = df[tissue_columns].values
    axes[0, 0].boxplot(volume_data, labels=tissue_names)
    axes[0, 0].set_title('各组织体积分布 (cm³)')
    axes[0, 0].set_ylabel('体积 (cm³)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 体积对比柱状图
    mean_volumes = df[tissue_columns].mean()
    axes[0, 1].bar(tissue_names, mean_volumes, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 1].set_title('各组织平均体积')
    axes[0, 1].set_ylabel('平均体积 (cm³)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 为每个柱子添加数值标签
    for i, v in enumerate(mean_volumes):
        axes[0, 1].text(i, v + max(mean_volumes)*0.01, f'{v:.1f}', 
                       ha='center', va='bottom', fontweight='bold')
    
    # T4高度分布
    axes[1, 0].hist(df['t4_height_slices'], bins=10, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_title('T4胸椎高度分布 (切片数)')
    axes[1, 0].set_xlabel('T4高度 (切片数)')
    axes[1, 0].set_ylabel('病人数量')
    
    # 相关性热力图
    correlation_cols = ['t4_height_slices'] + tissue_columns
    corr_matrix = df[correlation_cols].corr()
    im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1, 1].set_xticks(range(len(correlation_cols)))
    axes[1, 1].set_yticks(range(len(correlation_cols)))
    axes[1, 1].set_xticklabels([col.replace('_volume_cm3', '').replace('t4_height_slices', 'T4高度') 
                               for col in correlation_cols], rotation=45)
    axes[1, 1].set_yticklabels([col.replace('_volume_cm3', '').replace('t4_height_slices', 'T4高度') 
                               for col in correlation_cols])
    axes[1, 1].set_title('变量相关性矩阵')
    
    # 添加相关系数文本
    for i in range(len(correlation_cols)):
        for j in range(len(correlation_cols)):
            text = axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontweight='bold')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('t4_tissue_analysis.png', dpi=300, bbox_inches='tight')
        print("可视化图表已保存为: t4_tissue_analysis.png")
    
    plt.show()
    
    # 2. 创建详细的数据表格图
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 准备表格数据
    table_data = df[['patient_id', 't4_height_slices'] + tissue_columns].copy()
    table_data.columns = ['病人ID', 'T4高度(切片)', '皮下脂肪(cm³)', '纵隔脂肪(cm³)', '骨骼肌(cm³)']
    
    # 格式化数值
    for col in table_data.columns[2:]:
        table_data[col] = table_data[col].apply(lambda x: f'{x:.2f}')
    
    table = ax.table(cellText=table_data.values, colLabels=table_data.columns,
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # 设置表格样式
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('T4胸椎区域组织体积详细数据表', fontsize=14, fontweight='bold', pad=20)
    
    if save_plots:
        plt.savefig('t4_tissue_data_table.png', dpi=300, bbox_inches='tight')
        print("数据表格已保存为: t4_tissue_data_table.png")
    
    plt.show()

def generate_summary_report(df):
    """生成汇总报告"""
    if df is None or df.empty:
        return
    
    print("\n" + "="*60)
    print("T4胸椎区域组织体积分析汇总报告")
    print("="*60)
    
    print(f"总病人数: {len(df)}")
    print(f"T4高度范围: {df['t4_height_slices'].min()}-{df['t4_height_slices'].max()} 切片")
    print(f"T4平均高度: {df['t4_height_slices'].mean():.1f} ± {df['t4_height_slices'].std():.1f} 切片")
    
    # 显示裁剪后的尺寸信息
    if 'cropped_shape' in df.columns:
        print(f"\n裁剪后数据尺寸示例: {df['cropped_shape'].iloc[0]}")
        print("(宽度 x 高度 x T4高度切片数)")
    
    print("\n各组织体积统计 (cm³):")
    print("-" * 40)
    
    tissue_columns = [col for col in df.columns if col.endswith('_volume_cm3')]
    for col in tissue_columns:
        tissue_name = col.replace('_volume_cm3', '')
        mean_vol = df[col].mean()
        std_vol = df[col].std()
        min_vol = df[col].min()
        max_vol = df[col].max()
        
        print(f"{tissue_name}:")
        print(f"  平均值: {mean_vol:.2f} ± {std_vol:.2f} cm³")
        print(f"  范围: {min_vol:.2f} - {max_vol:.2f} cm³")
    
    print(f"\n输出文件:")
    print(f"- CSV结果文件: t4_tissue_volumes.csv")
    print(f"- 裁剪后组织掩膜: labelsTr_tissue_cropped/")
    print(f"- 裁剪后T4掩膜: labelsTr_T4_cropped/")
    print(f"- 裁剪后原始图像: imagesTr_cropped/")
    print(f"- 分析图表: t4_tissue_analysis.png")
    print(f"- 数据表格: t4_tissue_data_table.png")
    
    print("\n" + "="*60)

# 主函数
def main():
    """主执行函数"""
    print("开始处理T4胸椎区域组织体积计算...")
    print("-" * 50)
    
    # 处理所有病人数据
    results = process_all_patients()
    
    if not results:
        print("没有成功处理任何病人数据")
        return
    
    # 保存结果到CSV
    df = save_results_to_csv(results)
    
    # 生成可视化
    create_visualization(df)
    
    # 生成汇总报告
    generate_summary_report(df)
    
    print("\n处理完成!")
    return df

if __name__ == "__main__":
    # 运行主程序
    results_df = main()