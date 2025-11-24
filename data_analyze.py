import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class QuantitativeDataAnalyzer:
    """
    量化时序数据分析器
    """
    
    def __init__(self, train_data, val_data, output_dir="./"):
        # self.file_dir = file_dir
        # self.train_val_ratio = train_val_ratio
        # self.train_data = pd.DataFrame()
        # self.val_data = pd.DataFrame()
        self.train_data = train_data
        self.val_data = val_data
        self.comparison_results = None
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
   
    def comprehensive_data_analysis(self, key_columns=None):
        """
        对训练集和验证集进行全面的数据分析
        """
        if key_columns is None:
            # 默认重要特征
            key_columns = [
                'n_close', 'amount_delta', 'n_midprice', 
                'n_bid1', 'n_ask1', 'n_bsize1', 'n_asize1',
                'n_bid5', 'n_ask5'
            ]
        
        # 确保选择的特征在数据中存在
        available_features = [col for col in key_columns if col in self.train_data.columns]
        print(f"可用的重要特征: {available_features}")
        
        print("=" * 60)
        print("数据基本信息")
        print("=" * 60)
        
        # 基本信息
        print("\n训练集基本信息:")
        print(self.train_data[available_features].describe())
        
        print("\n验证集基本信息:")
        print(self.val_data[available_features].describe())
        
        # 检查缺失值
        print("\n训练集缺失值统计:")
        print(self.train_data[available_features].isnull().sum())
        
        print("\n验证集缺失值统计:")
        print(self.val_data[available_features].isnull().sum())
        
        # 数据分布比较分析
        print("\n" + "=" * 60)
        print("数据分布相似性分析")
        print("=" * 60)
        
        distribution_comparison = {}
        
        for col in available_features:
            # 移除可能的异常值用于更好的可视化
            train_clean = self.train_data[col].dropna()
            val_clean = self.val_data[col].dropna()
            
            # 统计检验 - KS检验
            ks_stat, ks_pvalue = ks_2samp(train_clean, val_clean)
            
            # 计算Wasserstein距离（推土机距离）
            wasserstein_dist = wasserstein_distance(train_clean, val_clean)
            
            # 计算均值差异
            mean_diff = abs(train_clean.mean() - val_clean.mean())
            mean_diff_ratio = mean_diff / (abs(train_clean.mean()) + 1e-8)
            
            distribution_comparison[col] = {
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'wasserstein_distance': wasserstein_dist,
                'mean_difference': mean_diff,
                'mean_difference_ratio': mean_diff_ratio,
                'similar_distribution': ks_pvalue > 0.05  # 通常使用0.05作为显著性水平
            }
        
        # 创建分布比较结果DataFrame
        self.comparison_results = pd.DataFrame(distribution_comparison).T
        self.comparison_results = self.comparison_results.sort_values('ks_statistic', ascending=False)
        
        print("\n特征分布相似性排序 (KS统计量越大，分布差异越大):")
        print(self.comparison_results[['ks_statistic', 'ks_pvalue', 'similar_distribution']])
        
        return self.comparison_results, available_features
    
    def plot_feature_distributions(self, features, n_cols=3, filename="feature_distributions.png"):
        """
        绘制训练集和验证集的特征分布对比图并保存
        """
        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(features):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # 清理数据
            train_vals = self.train_data[feature].dropna()
            val_vals = self.val_data[feature].dropna()
            
            # 绘制分布图
            sns.histplot(train_vals, ax=ax, label='训练集', alpha=0.7, 
                         color='blue', stat='density', kde=True)
            sns.histplot(val_vals, ax=ax, label='验证集', alpha=0.7, 
                         color='red', stat='density', kde=True)
            
            # 添加统计信息
            train_mean = train_vals.mean()
            val_mean = val_vals.mean()
            
            ax.axvline(train_mean, color='blue', linestyle='--', alpha=0.8, 
                      label=f'训练集均值: {train_mean:.4f}')
            ax.axvline(val_mean, color='red', linestyle='--', alpha=0.8,
                      label=f'验证集均值: {val_mean:.4f}')
            
            ax.set_title(f'{feature}分布对比\nKS统计量: {self.comparison_results.loc[feature, "ks_statistic"]:.4f}')
            ax.set_xlabel(feature)
            ax.set_ylabel('密度')
            ax.legend()
            
            # 添加分布相似性结论
            similarity = "分布相似" if self.comparison_results.loc[feature, "similar_distribution"] else "分布不同"
            ax.text(0.05, 0.95, f'结论: {similarity}', transform=ax.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
                    verticalalignment='top')
        
        # 隐藏多余的子图
        for idx in range(len(features), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"特征分布图已保存: {filename}")
    
    def plot_correlation_comparison(self, features, filename="correlation_comparison.png"):
        """
        比较训练集和验证集的相关性结构并保存
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 训练集相关性热图
        train_corr = self.train_data[features].corr()
        sns.heatmap(train_corr, ax=ax1, cmap='coolwarm', center=0, 
                    annot=True, fmt='.2f', square=True)
        ax1.set_title('训练集特征相关性矩阵')
        
        # 验证集相关性热图
        val_corr = self.val_data[features].corr()
        sns.heatmap(val_corr, ax=ax2, cmap='coolwarm', center=0, 
                    annot=True, fmt='.2f', square=True)
        ax2.set_title('验证集特征相关性矩阵')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"相关性对比图已保存: {filename}")
        
        # 计算相关性矩阵的差异
        corr_diff = np.abs(train_corr - val_corr)
        print("\n相关性矩阵平均绝对差异:", corr_diff.mean().mean())
        print("相关性矩阵最大差异:", corr_diff.max().max())
        
        return corr_diff
    
    def evaluate_split_quality(self):
        """
        评估数据划分的质量
        """
        if self.comparison_results is None:
            raise ValueError("请先执行数据分析")
            
        similar_features = self.comparison_results['similar_distribution'].sum()
        total_features = len(self.comparison_results)
        similarity_ratio = similar_features / total_features
        
        print("\n" + "=" * 60)
        print("数据划分质量评估")
        print("=" * 60)
        
        print(f"总特征数量: {total_features}")
        print(f"分布相似的特征数量: {similar_features}")
        print(f"分布相似性比例: {similarity_ratio:.2%}")
        
        # 评估标准
        if similarity_ratio >= 0.8:
            print("✅ 数据划分质量: 优秀 - 训练集和验证集分布高度一致")
        elif similarity_ratio >= 0.6:
            print("⚠️  数据划分质量: 良好 - 训练集和验证集分布基本一致")
        elif similarity_ratio >= 0.4:
            print("⚠️  数据划分质量: 一般 - 存在一定的分布差异")
        else:
            print("❌ 数据划分质量: 较差 - 训练集和验证集分布差异较大")
        
        # 显示分布差异最大的特征
        print("\n分布差异最大的5个特征:")
        print(self.comparison_results.head(5)[['ks_statistic', 'ks_pvalue']])
        # worst_features = self.comparison_results.nlargest(5, 'ks_statistic')

        self.comparison_results['ks_statistic'] = pd.to_numeric(
            self.comparison_results['ks_statistic'], 
            errors='coerce'  # 将无法转换的值设为NaN
        )
        valid_results = self.comparison_results.dropna(subset=['ks_statistic'])
        worst_features = valid_results.nlargest(5, 'ks_statistic')

        for feature, row in worst_features.iterrows():
            print(f"  {feature}: KS统计量={row['ks_statistic']:.4f}, p值={row['ks_pvalue']:.4f}")
        
        return similarity_ratio
    
    def analyze_temporal_properties(self, price_column='n_close'):
        """分析时间序列属性（严格异常值处理）"""
        try:
            # 计算收益率
            def safe_pct_change(prices):
                # 确保价格数据有效
                prices = prices.replace([np.inf, -np.inf], np.nan).dropna()
                if len(prices) < 2:
                    return pd.Series([], dtype=float)
                
                # 计算收益率
                returns = prices.pct_change().dropna()
                
                # 处理极端值
                returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
                
                # 使用IQR方法移除异常值
                if len(returns) > 0:
                    Q1 = returns.quantile(0.25)
                    Q3 = returns.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    
                    returns = returns[(returns >= lower_bound) & (returns <= upper_bound)]
                
                return returns
            
            train_returns = safe_pct_change(self.train_data[price_column])
            val_returns = safe_pct_change(self.val_data[price_column])
            
            if len(train_returns) == 0 or len(val_returns) == 0:
                print("警告: 没有有效的收益率数据，跳过时间序列分析")
                return
            
            # 设置合理的直方图范围
            all_returns = pd.concat([train_returns, val_returns])
            data_range = (all_returns.min(), all_returns.max())
            
            # 确保范围是有限的
            if not np.isfinite(data_range[0]) or not np.isfinite(data_range[1]):
                data_range = (-0.1, 0.1)  # 默认范围
            
            print(f"使用直方图范围: {data_range}")
            
            # 绘制直方图（指定有限的范围）
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            axes[0, 0].hist(train_returns, bins=50, alpha=0.7, label='训练集', 
                        density=True, color='blue', range=data_range)
            axes[0, 0].hist(val_returns, bins=50, alpha=0.7, label='验证集', 
                        density=True, color='red', range=data_range)
            axes[0, 0].set_title('收益率分布对比')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # ... 其余图表代码与上面相同
            
        except Exception as e:
            print(f"时间序列分析出错: {e}")


def analyze_quantitative_data(train_data, val_data, output_dir="./analysis_results",
                            key_columns=None, price_column='n_close'):
    """
    外部调用接口：执行完整的量化时序数据分析
    
    参数:
    - file_dir: 数据目录路径
    - train_val_ratio: 训练集比例
    - output_dir: 输出目录
    - key_columns: 重要特征列列表
    - price_column: 价格列名称（用于时间序列分析）
    
    返回:
    - analyzer: 分析器对象
    - comparison_results: 分布比较结果
    - available_features: 可用特征列表
    """
    
    # 创建分析器实例
    analyzer = QuantitativeDataAnalyzer(train_data, val_data, output_dir=output_dir)
    
    # 1. 加载和划分数据
    print("步骤1: 加载和划分数据...")
    print(f"train shape: {train_data.shape}, val shape: {val_data.shape}")
    # train_data, val_data = analyzer.load_and_split_data()
    
    # 2. 执行综合分析
    print("\n步骤2: 执行数据分布分析...")
    comparison_results, available_features = analyzer.comprehensive_data_analysis(key_columns)
    
    # 3. 绘制特征分布图
    print("\n步骤3: 绘制特征分布图...")
    # analyzer.plot_feature_distributions(available_features)
    
    # 4. 绘制相关性对比图
    print("\n步骤4: 绘制相关性对比图...")
    # analyzer.plot_correlation_comparison(available_features)
    
    # 5. 评估划分质量
    print("\n步骤5: 评估数据划分质量...")
    similarity_ratio = analyzer.evaluate_split_quality()
    
    # 6. 时间序列分析
    print("\n步骤6: 时间序列特性分析...")
    analyzer.analyze_temporal_properties(price_column)
    
    print("\n" + "=" * 60)
    print("数据分析完成！")
    print("=" * 60)
    
    return analyzer, comparison_results, available_features


# 使用示例
if __name__ == "__main__":
    # 定义重要特征
    important_features = [
        'n_close', 'amount_delta', 'n_midprice', 
        'n_bid1', 'n_ask1', 'n_bsize1', 'n_asize1',
        'n_bid5', 'n_ask5'
    ]
    
    # 调用外部接口函数
    analyzer, results, features = analyze_quantitative_data(
        file_dir="./data",
        train_val_ratio=0.8,
        output_dir="./analysis_results",
        key_columns=important_features,
        price_column='n_close'
    )
    
    # 可以继续使用analyzer对象进行其他操作
    print(f"\n分析完成！共分析 {len(features)} 个特征")
    print(f"所有图表已保存到: {analyzer.output_dir}")