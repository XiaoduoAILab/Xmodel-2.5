import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 1. Stable 阶段数据 ---------------------------------------------------
stable_labels = [
    'ultrafineweb-en', 'starcoder', 'ultrafineweb-zh',
    'reddit', 'pes2o', 'arxiv', 'books', 'wiki',
    'stackexchange', 'tulu_flan', 'algebraic-stack',
    'open-web-math', 'megawika'
]
stable_sizes = np.array([41.00, 26.00, 20.00, 3.91, 2.85, 1.33, 1.00, 1.00,
                         0.90, 0.78, 0.50, 0.50, 0.23])

# 2. Decay 阶段数据（已换算成 %） ---------------------------------------
decay_raw = {
    'ultrafineweb-en': 0.18725,
    'ultrafineweb-zh': 0.03716,
    'starcoder': 0.02209,  
    'books': 0.00287,
    'algebraic-stack-train': 0.01722,
    'open-web-math-train': 0.01722,
    'wiki': 0.01147,     
    'stackexchange': 0.00900,
    'reddit': 0.00900,
    'megawika': 0.01801,
    'SFT_mixed': 0.66870,
}
decay_labels = list(decay_raw.keys())
decay_sizes  = np.array(list(decay_raw.values())) * 100   # 转为百分比

# 3. 画图 --------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7),
                               subplot_kw=dict(aspect='equal'))

# 使用更专业的配色方案
colors1 = plt.cm.tab20(np.linspace(0, 1, len(stable_labels)))
colors2 = plt.cm.Set3(np.linspace(0, 1, len(decay_labels)))

# 自定义百分比显示函数 - 只显示大于1%的部分
def custom_autopct(pct, allvals):
    if pct > 1:  # 只显示大于1%的百分比
        return f'{pct:.1f}%'
    else:
        return ''

# 左图：Stable
wedges1, texts1, autotexts1 = ax1.pie(stable_sizes, labels=None,
                                     autopct=lambda pct: custom_autopct(pct, stable_sizes),
                                     startangle=90, colors=colors1,
                                     pctdistance=0.8, labeldistance=1.1)
ax1.set_title('Stable Stage Data Proportion', fontsize=14, fontweight='bold', pad=20)

# 调整Stable图的标签位置
texts1 = [t for t in autotexts1 if t.get_text() != '']
if texts1:
    adjust_text(texts1, ax=ax1,
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, alpha=0.7),
                force_points=2.0, force_text=0.5, expand_points=(1.5, 1.5))

# 右图：Decay
wedges2, texts2, autotexts2 = ax2.pie(decay_sizes, labels=None,
                                     autopct=lambda pct: custom_autopct(pct, decay_sizes),
                                     startangle=90, colors=colors2,
                                     pctdistance=0.8, labeldistance=1.1)
ax2.set_title('Decay Stage Data Proportion', fontsize=14, fontweight='bold', pad=20)

# 调整Decay图的标签位置
texts2 = [t for t in autotexts2 if t.get_text() != '']
if texts2:
    adjust_text(texts2, ax=ax2,
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, alpha=0.7),
                force_points=2.0, force_text=0.5, expand_points=(1.5, 1.5))

# 创建图例（放在图表外部，避免重叠）
legend_labels1 = [f"{label}: {size:.2f}%" for label, size in zip(stable_labels, stable_sizes)]
legend_labels2 = [f"{label}: {size:.2f}%" for label, size in zip(decay_labels, decay_sizes)]

# 将图例放在图表下方
fig.legend(wedges1, legend_labels1,
          title='Stable Sources',
          loc='lower left', bbox_to_anchor=(0.15, 0.05),
          ncol=2, fontsize=8)

fig.legend(wedges2, legend_labels2,
          title='Decay Sources',
          loc='lower right', bbox_to_anchor=(0.85, 0.05),
          ncol=2, fontsize=8)

# 调整布局，为图例留出空间
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)  # 为底部图例留出空间

# 4. 保存 --------------------------------------------------------------
plt.savefig('stable_vs_decay_enhanced.pdf', dpi=600, bbox_inches='tight')
plt.savefig('stable_vs_decay_enhanced.png', dpi=600, bbox_inches='tight')
plt.show()