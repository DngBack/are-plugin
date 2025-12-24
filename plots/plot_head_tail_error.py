import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# --- 1. Load dữ liệu từ JSON files ---
def load_head_tail_error_data(json_path: str, method_name: str):
    """Load dữ liệu Tail Error - Head Error từ JSON file."""
    json_file = Path(json_path)
    if not json_file.exists():
        print(f"Warning: {json_path} not found. Skipping {method_name} data.")
        return None, None
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rejection_rates = []
    tail_minus_head_errors = []
    
    # Extract từ results_per_cost (cho balanced methods)
    if 'results_per_cost' in data:
        for result in data['results_per_cost']:
            test_metrics = result.get('test_metrics', {})
            group_errors = test_metrics.get('group_errors', [])
            
            if len(group_errors) >= 2:
                head_error = float(group_errors[0])
                tail_error = float(group_errors[1])
                tail_minus_head = tail_error - head_error
                
                # Lấy rejection_rate từ test_metrics hoặc tính từ coverage
                rejection_rate = test_metrics.get('rejection_rate', None)
                if rejection_rate is None:
                    coverage = test_metrics.get('coverage', 1.0)
                    rejection_rate = 1.0 - coverage
                
                rejection_rates.append(float(rejection_rate))
                tail_minus_head_errors.append(tail_minus_head)
    
    # Extract từ results_per_point (cho worst methods)
    elif 'results_per_point' in data:
        for result in data['results_per_point']:
            test_metrics = result.get('test_metrics', {})
            group_errors = test_metrics.get('group_errors', [])
            
            if len(group_errors) >= 2:
                head_error = float(group_errors[0])
                tail_error = float(group_errors[1])
                tail_minus_head = tail_error - head_error
                
                # Lấy rejection_rate từ target_rejection hoặc tính từ coverage
                rejection_rate = result.get('target_rejection', None)
                if rejection_rate is None:
                    coverage = test_metrics.get('coverage', 1.0)
                    rejection_rate = 1.0 - coverage
                
                rejection_rates.append(float(rejection_rate))
                tail_minus_head_errors.append(tail_minus_head)
    
    if len(rejection_rates) > 0:
        # Sort theo rejection_rate
        sorted_indices = np.argsort(rejection_rates)
        rejection_rates = np.array(rejection_rates)[sorted_indices]
        tail_minus_head_errors = np.array(tail_minus_head_errors)[sorted_indices]
        return rejection_rates, tail_minus_head_errors
    
    return None, None

# Load dữ liệu từ các file JSON
ce_only_balanced_rejections, ce_only_balanced_errors = load_head_tail_error_data(
    './results/ltr_plugin/cifar100_lt_if100/ltr_plugin_ce_only_balanced.json',
    'CE Only (Balanced)'
)

ce_only_worst_rejections, ce_only_worst_errors = load_head_tail_error_data(
    './results/ltr_plugin/cifar100_lt_if100/ltr_plugin_ce_only_worst.json',
    'CE Only (Worst)'
)

gating_balanced_rejections, gating_balanced_errors = load_head_tail_error_data(
    './results/ltr_plugin/cifar100_lt_if100/ltr_plugin_gating_balanced.json',
    'ARE (Balanced)'
)

gating_worst_rejections, gating_worst_errors = load_head_tail_error_data(
    './results/ltr_plugin/cifar100_lt_if100/ltr_plugin_gating_worst.json',
    'ARE (Worst)'
)

# In thông tin để xác nhận đã load dữ liệu
print("=== Data Loading Summary ===")
if ce_only_balanced_rejections is not None:
    print(f"✓ CE Only (Balanced): {len(ce_only_balanced_rejections)} points")
if ce_only_worst_rejections is not None:
    print(f"✓ CE Only (Worst): {len(ce_only_worst_rejections)} points")
if gating_balanced_rejections is not None:
    print(f"✓ ARE (Balanced): {len(gating_balanced_rejections)} points")
if gating_worst_rejections is not None:
    print(f"✓ ARE (Worst): {len(gating_worst_rejections)} points")

# --- 2. Dữ liệu Chow's rule (từ paper) ---
chow_rejections = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
chow_errors = np.array([0.48, 0.50, 0.53, 0.56, 0.59, 0.60, 0.60, 0.57, 0.54])

# --- 3. Chuẩn bị dữ liệu cho DataFrame ---
data_dict = {
    'Proportion of Rejections': chow_rejections,
    "Chow's rule": chow_errors,
}

# Thêm dữ liệu CE Only (Balanced)
if ce_only_balanced_rejections is not None and ce_only_balanced_errors is not None:
    print(f"\n✓ CE Only (Balanced) data range:")
    print(f"  Rejection rates: [{float(ce_only_balanced_rejections.min()):.3f}, {float(ce_only_balanced_rejections.max()):.3f}]")
    print(f"  Tail - Head errors: [{float(ce_only_balanced_errors.min()):.3f}, {float(ce_only_balanced_errors.max()):.3f}]")

# Thêm dữ liệu CE Only (Worst)
if ce_only_worst_rejections is not None and ce_only_worst_errors is not None:
    print(f"\n✓ CE Only (Worst) data range:")
    print(f"  Rejection rates: [{float(ce_only_worst_rejections.min()):.3f}, {float(ce_only_worst_rejections.max()):.3f}]")
    print(f"  Tail - Head errors: [{float(ce_only_worst_errors.min()):.3f}, {float(ce_only_worst_errors.max()):.3f}]")

# Thêm dữ liệu ARE (Balanced)
if gating_balanced_rejections is not None and gating_balanced_errors is not None:
    print(f"\n✓ ARE (Balanced) data range:")
    print(f"  Rejection rates: [{float(gating_balanced_rejections.min()):.3f}, {float(gating_balanced_rejections.max()):.3f}]")
    print(f"  Tail - Head errors: [{float(gating_balanced_errors.min()):.3f}, {float(gating_balanced_errors.max()):.3f}]")

# Thêm dữ liệu ARE (Worst)
if gating_worst_rejections is not None and gating_worst_errors is not None:
    print(f"\n✓ ARE (Worst) data range:")
    print(f"  Rejection rates: [{float(gating_worst_rejections.min()):.3f}, {float(gating_worst_rejections.max()):.3f}]")
    print(f"  Tail - Head errors: [{float(gating_worst_errors.min()):.3f}, {float(gating_worst_errors.max()):.3f}]")

# Chuyển dữ liệu Chow sang DataFrame
df_chow = pd.DataFrame(data_dict)
df_chow_melted = df_chow.melt(
    id_vars=['Proportion of Rejections'],
    var_name='Method',
    value_name='Tail Error - Head Error'
)

# Tạo DataFrame cho các phương pháp từ JSON
dfs_list = [df_chow_melted]

if ce_only_balanced_rejections is not None and ce_only_balanced_errors is not None:
    df_ce_bal = pd.DataFrame({
        'Proportion of Rejections': ce_only_balanced_rejections,
        'CE Only (Balanced)': ce_only_balanced_errors
    })
    df_ce_bal_melted = df_ce_bal.melt(
        id_vars=['Proportion of Rejections'],
        var_name='Method',
        value_name='Tail Error - Head Error'
    )
    dfs_list.append(df_ce_bal_melted)

if ce_only_worst_rejections is not None and ce_only_worst_errors is not None:
    df_ce_worst = pd.DataFrame({
        'Proportion of Rejections': ce_only_worst_rejections,
        'CE Only (Worst)': ce_only_worst_errors
    })
    df_ce_worst_melted = df_ce_worst.melt(
        id_vars=['Proportion of Rejections'],
        var_name='Method',
        value_name='Tail Error - Head Error'
    )
    dfs_list.append(df_ce_worst_melted)

if gating_balanced_rejections is not None and gating_balanced_errors is not None:
    df_gating_bal = pd.DataFrame({
        'Proportion of Rejections': gating_balanced_rejections,
        'ARE (Balanced)': gating_balanced_errors
    })
    df_gating_bal_melted = df_gating_bal.melt(
        id_vars=['Proportion of Rejections'],
        var_name='Method',
        value_name='Tail Error - Head Error'
    )
    dfs_list.append(df_gating_bal_melted)

if gating_worst_rejections is not None and gating_worst_errors is not None:
    df_gating_worst = pd.DataFrame({
        'Proportion of Rejections': gating_worst_rejections,
        'ARE (Worst)': gating_worst_errors
    })
    df_gating_worst_melted = df_gating_worst.melt(
        id_vars=['Proportion of Rejections'],
        var_name='Method',
        value_name='Tail Error - Head Error'
    )
    dfs_list.append(df_gating_worst_melted)

# Gộp tất cả dữ liệu
df = pd.concat(dfs_list, ignore_index=True)

# --- 4. Thiết lập Biểu đồ (Plotting Setup) ---
sns.set_theme(style="darkgrid")

# Định nghĩa các marker và màu sắc tùy chỉnh
markers = {
    "Chow's rule": 'o',
    'CE Only (Balanced)': 's',
    'CE Only (Worst)': 's',
    'ARE (Balanced)': 'D',
    'ARE (Worst)': 'D',
}

# Định nghĩa màu sắc
colors = {
    "Chow's rule": None,  # Để seaborn tự chọn
    'CE Only (Balanced)': '#2E86AB',  # Xanh dương
    'CE Only (Worst)': '#A23B72',  # Tím hồng
    'ARE (Balanced)': '#F18F01',  # Cam
    'ARE (Worst)': '#9370DB',  # Tím
}

# Định nghĩa line styles - tất cả nét liền
linestyles = {
    "Chow's rule": '-',
    'CE Only (Balanced)': '-',
    'CE Only (Worst)': '-',
    'ARE (Balanced)': '-',
    'ARE (Worst)': '-',
}

plt.figure(figsize=(10, 7))

# --- 5. Vẽ Biểu đồ (The Plot) ---
for method in df['Method'].unique():
    subset = df[df['Method'] == method].sort_values('Proportion of Rejections')
    
    # Lấy marker, màu và line style
    marker = markers.get(method, 'o')
    color = colors.get(method, None)
    linestyle = linestyles.get(method, '-')
    
    # Vẽ đường
    sns.lineplot(
        data=subset,
        x='Proportion of Rejections',
        y='Tail Error - Head Error',
        label=method,  # Thêm label cho legend
        marker=marker,
        color=color,
        linestyle=linestyle,
        markersize=10,
        markeredgecolor='white' if method in ["Chow's rule", 'CE Only (Balanced)', 'CE Only (Worst)', 'ARE (Balanced)', 'ARE (Worst)'] else None,
        markeredgewidth=2 if method in ["Chow's rule", 'CE Only (Balanced)', 'CE Only (Worst)', 'ARE (Balanced)', 'ARE (Worst)'] else None,
        linewidth=4.5,  # Tăng độ dày nét vẽ
        zorder=3,
        err_style='band',
        ci=None,
        dashes=False,
    )

# --- 6. Tùy chỉnh Cuối cùng (Final Customizations) ---

# Đặt giới hạn trục Y (có thể cần điều chỉnh dựa trên dữ liệu thực tế)
all_errors = df['Tail Error - Head Error'].dropna()
if len(all_errors) > 0:
    y_min = max(0.0, float(all_errors.min()) - 0.05)
    y_max = float(all_errors.max()) + 0.05
    plt.ylim(y_min, y_max)
else:
    plt.ylim(0.0, 0.7)

# Đặt giới hạn trục X
plt.xlim(0.0, 0.8)

# Đặt tên trục
plt.xlabel('Proportion of Rejections', fontsize=18)
plt.ylabel('Tail Error - Head Error', fontsize=18)

# Tăng kích thước font cho ticks
plt.tick_params(axis='both', which='major', labelsize=14)

# Thêm legend
plt.legend(
    title=None,
    loc='best',
    fontsize=14,
    frameon=True,
    edgecolor='black'
)

# Hiển thị biểu đồ
plt.grid(True, axis='both', linestyle='-', alpha=0.5)
plt.tight_layout()

# Lưu biểu đồ
output_path = './results/paper_figures/head_tail_error_comparison.png'
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved plot to: {output_path}")

plt.show()

