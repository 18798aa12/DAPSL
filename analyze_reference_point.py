#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CatBoost参考点分析工具
用于分析CatBoost模型的输出范围并推荐合适的参考点
"""

import numpy as np
import pickle
import pandas as pd
import glob
import os

def analyze_model_output_range():
    """分析CatBoost模型的输出范围"""
    
    print("🔍 分析CatBoost模型输出范围...")
    
    try:
        # 加载模型
        model_path = '0624_model/catboost.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("✅ 模型加载成功")

        # 生成大量随机样本
        n_samples = 2000
        np.random.seed(42)  # 确保可重复性
        X_test = np.random.rand(n_samples, 13)

        # 获取预测值
        predictions = model.predict(X_test)
        print(f"✅ 成功预测 {len(predictions)} 个样本")

        print(f"\n📊 氨排放强度预测统计:")
        print(f"  最小值: {predictions.min():.6f}")
        print(f"  最大值: {predictions.max():.6f}")
        print(f"  均值: {predictions.mean():.6f}")
        print(f"  标准差: {predictions.std():.6f}")
        
        percentiles = [25, 50, 75, 90, 95, 99]
        print(f"  分位数:")
        for p in percentiles:
            val = np.percentile(predictions, p)
            print(f"    {p:2d}%: {val:.6f}")

        return predictions.min(), predictions.max(), predictions.mean(), predictions.std()
        
    except Exception as e:
        print(f"❌ 模型分析失败: {e}")
        return None, None, None, None

def analyze_experiment_results():
    """分析实验结果中的目标函数值范围"""
    
    print("\n🔬 分析实验结果中的目标函数值...")
    
    # 查找最新的实验结果
    results_dirs = glob.glob("results/experiment_*")
    if not results_dirs:
        print("❌ 未找到实验结果目录")
        return None, None
    
    latest_dir = max(results_dirs)
    pareto_dir = f"{latest_dir}/pareto_fronts/catboost_min"
    
    if not os.path.exists(pareto_dir):
        print(f"❌ 未找到Pareto前沿数据: {pareto_dir}")
        return None, None
    
    # 读取所有CSV文件
    csv_files = glob.glob(f"{pareto_dir}/*.csv")
    print(f"📁 找到 {len(csv_files)} 个Pareto前沿文件")
    
    all_obj1 = []
    all_obj2 = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if not df.empty and 'Obj_1' in df.columns:
                all_obj1.extend(df['Obj_1'].tolist())
                all_obj2.extend(df['Obj_2'].tolist())
        except Exception as e:
            print(f"⚠️ 读取文件失败 {file}: {e}")
            continue
    
    if not all_obj1:
        print("❌ 未找到有效的目标函数值")
        return None, None
    
    all_obj1 = np.array(all_obj1)
    all_obj2 = np.array(all_obj2)
    
    print(f"\n🎯 实验中的目标函数值分析 (基于 {len(all_obj1)} 个解):")
    print(f"目标1 (氨排放强度):")
    print(f"  范围: [{all_obj1.min():.6f}, {all_obj1.max():.6f}]")
    print(f"  均值: {all_obj1.mean():.6f}")
    print(f"  标准差: {all_obj1.std():.6f}")
    
    print(f"\n目标2 (氨排放强度的一半):")
    print(f"  范围: [{all_obj2.min():.6f}, {all_obj2.max():.6f}]")
    print(f"  均值: {all_obj2.mean():.6f}")
    print(f"  标准差: {all_obj2.std():.6f}")
    
    return all_obj1, all_obj2

def recommend_reference_point(model_max=None, exp_obj1=None, exp_obj2=None):
    """推荐最佳参考点"""
    
    print("\n💡 参考点推荐分析:")
    print("="*50)
    
    current_ref = [5.0, 2.5]
    print(f"当前参考点: {current_ref}")
    
    # 确定最佳参考点
    recommended_ref = None
    confidence = "低"
    
    # 优先使用实验结果，因为更准确
    if exp_obj1 is not None and exp_obj2 is not None:
        obj1_max = exp_obj1.max()
        obj2_max = exp_obj2.max()
        
        # 使用30%的安全边界，这是经验上的最佳平衡
        margin = 1.3
        rec_obj1 = obj1_max * margin
        rec_obj2 = obj2_max * margin
        
        recommended_ref = [round(rec_obj1, 1), round(rec_obj2, 1)]
        confidence = "高"
        
        print(f"� 基于实验数据分析:")
        print(f"  目标1最大值: {obj1_max:.6f}")
        print(f"  目标2最大值: {obj2_max:.6f}")
        
    # 如果没有实验结果，使用模型预测
    elif model_max is not None:
        margin = 1.3
        rec_obj1 = model_max * margin
        rec_obj2 = model_max * 0.5 * margin
        
        recommended_ref = [round(rec_obj1, 1), round(rec_obj2, 1)]
        confidence = "中"
        
        print(f"📈 基于模型预测分析:")
        print(f"  预测最大值: {model_max:.6f}")
    
    # 如果都没有，使用保守估计
    else:
        recommended_ref = [0.5, 0.25]  # 基于经验的保守估计
        confidence = "低"
        print("⚠️ 无法获取数据，使用保守估计")
    
    return recommended_ref, confidence

def main():
    """主函数"""
    print("🎯 CatBoost参考点智能推荐工具")
    print("="*60)
    
    # 分析模型输出范围
    model_min, model_max, model_mean, model_std = analyze_model_output_range()
    
    # 分析实验结果
    exp_obj1, exp_obj2 = analyze_experiment_results()
    
    # 获取推荐参考点
    recommended_ref, confidence = recommend_reference_point(model_max, exp_obj1, exp_obj2)
    
    # 输出最终推荐
    print("\n" + "="*60)
    print("🏆 最终推荐结果")
    print("="*60)
    print(f"� 推荐参考点: {recommended_ref}")
    print(f"🎯 推荐置信度: {confidence}")
    
    current_ref = [5.0, 2.5]
    if recommended_ref != current_ref:
        print(f"📝 建议修改 run.py 中的参考点:")
        print(f"   当前: 'catboost_min': {current_ref}")
        print(f"   推荐: 'catboost_min': {recommended_ref}")
        
        # 计算改进预期
        if exp_obj1 is not None:
            current_margin1 = current_ref[0] / exp_obj1.max() if exp_obj1.max() > 0 else float('inf')
            recommended_margin1 = recommended_ref[0] / exp_obj1.max() if exp_obj1.max() > 0 else float('inf')
            
            if current_margin1 > 5:
                print(f"💡 当前参考点可能过于保守 ({current_margin1:.1f}倍)")
                print(f"   新参考点更合理 ({recommended_margin1:.1f}倍)")
            elif current_margin1 < 1.2:
                print(f"⚠️ 当前参考点可能过小 ({current_margin1:.1f}倍)")
                print(f"   新参考点更安全 ({recommended_margin1:.1f}倍)")
    else:
        print("✅ 当前参考点已经是最佳选择！")
    
    print(f"\n📖 使用说明:")
    print(f"   将推荐的参考点 {recommended_ref} 复制到 run.py 第242行")
    print(f"   替换: dic = {{'catboost_min': {recommended_ref}}}")
    
    print("\n✅ 分析完成!")
    return recommended_ref

if __name__ == "__main__":
    recommended_ref = main()
    
    # 提供直接可用的代码片段
    print("\n" + "="*60)
    print("📋 代码更新建议")
    print("="*60)
    print("请将以下代码复制到 run.py 的第241-243行:")
    print("-" * 40)
    print("dic = {")
    print(f"    'catboost_min': {recommended_ref},")
    print("    # 'rf_max': [-5.0, -5.0]  # 最小化问题的参考点 (正值)")
    print("}")
    print("-" * 40)
    print("✨ 修改完成后即可运行优化实验！")
