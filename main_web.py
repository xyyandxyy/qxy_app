from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MultipleLocator
from flask import Flask, render_template, request, send_file, redirect, url_for, flash, jsonify
import io
import base64
import matplotlib
import os
import uuid
import sys
import webbrowser
from werkzeug.utils import secure_filename
from matplotlib import font_manager
matplotlib.use('Agg')

# 导入资源路径处理函数
from setup import resource_path, setup_app

# 注册阿里巴巴普惠体
font_path = setup_app()  # 获取字体路径
font_prop = font_manager.FontProperties(fname=font_path)
font_manager.fontManager.addfont(font_path)
alibaba_font = font_manager.FontProperties(fname=font_path).get_name()

# 设置字体
plt.rcParams['font.sans-serif'] = [alibaba_font, 'SimHei', 'Arial Unicode MS']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn默认使用该字体
sns.set(font=alibaba_font, font_scale=1)

app = Flask(__name__)
app.secret_key = str(uuid.uuid4())  # 为了使用flash消息功能
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'xlsx', 'csv'}

# 创建上传文件夹
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

processed_df = None
column_info = {}
data_summary = {}
current_file = None
column_id_map = {}  # 列名到ID的映射
id_column_map = {}  # ID到列名的映射

def detect_column_type(series):
    """智能检测列的数据类型"""
    # 去除空值
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return 'empty'
    
    # 检测数值类型
    numeric_series = pd.to_numeric(clean_series, errors='coerce')
    numeric_ratio = numeric_series.notna().sum() / len(clean_series)
    
    if numeric_ratio > 0.8:  # 80%以上是数值
        # 检查是否为整数类型（修复语法错误）
        if numeric_series.dtype == 'int64':
            return 'integer'
        else:
            # 检查浮点数是否都是整数值（修复逻辑错误）
            non_null_numeric = numeric_series.dropna()
            if len(non_null_numeric) > 0 and all(non_null_numeric == non_null_numeric.astype(int)):
                return 'integer'
            else:
                return 'float'
    
    # 检测日期类型
    date_series = pd.to_datetime(clean_series, errors='coerce')
    date_ratio = date_series.notna().sum() / len(clean_series)
    if date_ratio > 0.8:
        return 'datetime'
    
    # 检测布尔类型（是/否）
    unique_values = set(clean_series.astype(str).unique())
    bool_keywords = [{'是', '否'}, {'True', 'False'}, {'1', '0'}, {'有', '无'}]
    for keywords in bool_keywords:
        if unique_values.issubset(keywords):
            return 'boolean'
    
    # 检测分类类型
    unique_count = len(clean_series.unique())
    total_count = len(clean_series)
    
    if unique_count <= 10 or unique_count / total_count < 0.1:
        return 'categorical'
    
    return 'text'

def analyze_data_structure(df):
    """分析数据结构"""
    global column_info, data_summary, column_id_map, id_column_map
    
    column_info = {}
    column_id_map = {}
    id_column_map = {}
    
    # 为每个列名分配一个唯一的数字ID
    for i, col in enumerate(df.columns):
        col_id = i + 1  # 从1开始的ID
        column_id_map[col] = col_id
        id_column_map[col_id] = col
        
        col_type = detect_column_type(df[col])
        unique_count = df[col].nunique()
        null_count = df[col].isnull().sum()
        
        column_info[col] = {
            'type': col_type,
            'unique_count': unique_count,
            'null_count': null_count,
            'null_ratio': null_count / len(df),
            'sample_values': df[col].dropna().head(3).tolist(),
            'id': col_id  # 存储列ID
        }
    
    # 生成数据摘要
    data_summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': [col for col, info in column_info.items() if info['type'] in ['integer', 'float']],
        'categorical_columns': [col for col, info in column_info.items() if info['type'] == 'categorical'],
        'boolean_columns': [col for col, info in column_info.items() if info['type'] == 'boolean'],
        'datetime_columns': [col for col, info in column_info.items() if info['type'] == 'datetime'],
        'text_columns': [col for col, info in column_info.items() if info['type'] == 'text']
    }

def clean_and_process_data(df):
    """清理和处理数据"""
    # 移除完全空的行
    df_clean = df.dropna(how='all').copy()
    
    # 处理布尔列
    for col, info in column_info.items():
        if info['type'] == 'boolean' and col in df_clean.columns:
            # 标准化布尔值
            df_clean[col] = df_clean[col].astype(str).str.strip()
            bool_map = {'是': '是', '否': '否', 'True': '是', 'False': '否', 
                       '1': '是', '0': '否', '有': '是', '无': '否'}
            df_clean[col] = df_clean[col].map(bool_map)
    
    # 处理数值列
    for col in data_summary['numeric_columns']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # 处理日期列
    for col in data_summary['datetime_columns']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    return df_clean

def plot_to_base64():
    """将matplotlib图表转换为base64字符串"""
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url, img

def generate_chart(column, chart_type, save_path=None, exclude_zeros=False, y_step_size=None, x_step_size=None, y_min=None, y_max=None, x_min=None, x_max=None, chart_color=None, y_label=None, x_label=None, chart_title=None):
    """根据列和图表类型生成图表
    
    参数:
        column: 要生成图表的列名
        chart_type: 图表类型
        save_path: 如果提供，则将图表保存到该路径
        exclude_zeros: 如果为True，则排除所有数值为0的数据点
        y_step_size: 纵轴刻度的步长
        x_step_size: 横轴刻度的步长
        y_min: 纵轴最小值
        y_max: 纵轴最大值
        x_min: 横轴最小值
        x_max: 横轴最大值
        chart_color: 图表颜色
        y_label: 自定义纵轴名称
        x_label: 自定义横轴名称
        chart_title: 自定义图表标题
    """
    col_info = column_info[column]
    data = processed_df[column].dropna()
    
    # 如果是数值列且需要排除零值
    if exclude_zeros and col_info['type'] in ['integer', 'float']:
        data = data[data != 0]
    
    # 设置seaborn样式
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # 使用自定义配色方案
    if chart_color:
        colors = [chart_color]
    else:
        colors = sns.color_palette("husl", 8)
    
    # 确保使用阿里巴巴普惠体
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': [alibaba_font, 'SimHei']})
    
    if col_info['type'] in ['categorical', 'boolean']:
        value_counts = data.value_counts()
        
        if chart_type == 'pie':
            def autopct_format(values):
                def my_format(pct):
                    total = sum(values)
                    val = int(round(pct*total/100.0))
                    return f'{val} ({pct:.1f}%)'
                return my_format
            
            plt.pie(value_counts.values, labels=value_counts.index, 
                   autopct=autopct_format(value_counts.values),
                   colors=colors, shadow=True, startangle=90,
                   wedgeprops={'edgecolor': 'w', 'linewidth': 1})
            plt.title(chart_title if chart_title else f'{column} - 饼图分布', fontsize=16, fontweight='bold', fontproperties=font_prop)
            
        elif chart_type == 'bar':
            # 使用seaborn的barplot
            ax = sns.barplot(x=value_counts.index, y=value_counts.values, palette=colors)
            plt.xticks(rotation=45, ha='right')
            
            # 添加数值标签
            for i, p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(p.get_x() + p.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', fontweight='bold')
            
            plt.title(chart_title if chart_title else f'{column} - 柱状图分布', fontsize=16, fontweight='bold', fontproperties=font_prop)
            plt.ylabel(y_label if y_label else '数量', fontsize=12, fontproperties=font_prop)
            plt.xlabel(x_label if x_label else column, fontsize=12, fontproperties=font_prop)
    
    elif col_info['type'] in ['integer', 'float']:
        if chart_type == 'histogram':
            # 使用seaborn的histplot
            ax = sns.histplot(data, bins=20, kde=True, color=colors[0], edgecolor='white', linewidth=1)
            counts, bins = np.histogram(data, bins=20)
            
            # 保存bin数据供后续使用
            hist_data = {'counts': counts, 'bins': bins}
            
            plt.title(chart_title if chart_title else f'{column} - 直方图分布', fontsize=16, fontweight='bold', fontproperties=font_prop)
            plt.xlabel(x_label if x_label else column, fontsize=12, fontproperties=font_prop)
            plt.ylabel(y_label if y_label else '频次', fontsize=12, fontproperties=font_prop)
            
        elif chart_type == 'line':
            sorted_data = data.sort_values()
            # 使用seaborn的lineplot
            ax = sns.lineplot(x=range(len(sorted_data)), y=sorted_data.values, marker='o', markersize=5, alpha=0.9, color=colors[1])
            plt.title(chart_title if chart_title else f'{column} - 折线图', fontsize=16, fontweight='bold', fontproperties=font_prop)
            plt.xlabel(x_label if x_label else '索引', fontsize=12, fontproperties=font_prop)
            plt.ylabel(y_label if y_label else column, fontsize=12, fontproperties=font_prop)
            
        elif chart_type == 'bar':
            # 对于数值类型的柱状图，显示值的分布
            value_counts = data.value_counts().head(20)  # 只显示前20个最常见的值
            
            # 使用seaborn的barplot
            ax = sns.barplot(x=value_counts.index, y=value_counts.values, palette=colors)
            plt.xticks(rotation=45, ha='right')
            
            # 添加数值标签
            for i, p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(p.get_x() + p.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', fontweight='bold')
            
            plt.title(chart_title if chart_title else f'{column} - 值分布柱状图', fontsize=16, fontweight='bold', fontproperties=font_prop)
            plt.xlabel(x_label if x_label else column, fontsize=12, fontproperties=font_prop)
            plt.ylabel(y_label if y_label else '频次', fontsize=12, fontproperties=font_prop)
    
    # 添加网格线美化
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置坐标轴范围和步长
    ax = plt.gca()
    # 设置纵轴步长
    if y_step_size and y_step_size.strip() and col_info['type'] in ['integer', 'float']:
        ax.yaxis.set_major_locator(plt.MultipleLocator(float(y_step_size)))
    # 设置横轴步长
    if x_step_size and x_step_size.strip() and chart_type != 'pie' and col_info['type'] in ['integer', 'float']:
        ax.xaxis.set_major_locator(plt.MultipleLocator(float(x_step_size)))
    
    # 安全转换值，检查None和空字符串
    def safe_float(value):
        if value is None or value == '' or not value.strip():
            return None
        return float(value)
    
    y_min_float = safe_float(y_min)
    y_max_float = safe_float(y_max)
    x_min_float = safe_float(x_min)
    x_max_float = safe_float(x_max)
    
    if y_min_float is not None or y_max_float is not None:
        plt.ylim(bottom=y_min_float, top=y_max_float)
    
    if (x_min_float is not None or x_max_float is not None) and chart_type != 'pie':
        plt.xlim(left=x_min_float, right=x_max_float)
        
    # 在设置坐标轴范围后添加直方图标签
    if chart_type == 'histogram' and col_info['type'] in ['integer', 'float']:
        # 获取当前的x轴范围
        x_min_current, x_max_current = plt.xlim()
        y_min_current, y_max_current = plt.ylim()
        
        # 访问histogram数据
        if 'hist_data' in locals():
            counts = hist_data['counts']
            bins = hist_data['bins']
            
            # 添加数值标签，仅在可见区域内
            for i, count in enumerate(counts):
                bin_center = bins[i] + (bins[i+1] - bins[i])/2
                if count > 0 and x_min_current <= bin_center <= x_max_current and count <= y_max_current:
                    # 确保标签不会超出图表顶部
                    label_y_pos = min(count, y_max_current * 0.95)  # 留一些边距
                    plt.text(bin_center, label_y_pos, 
                            f'{int(count)}', ha='center', va='bottom', fontsize=9)
    
    # 增强图表边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    
    # 如果需要保存到文件
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)
    
    # 返回base64编码的图像
    plot_url, img_buffer = plot_to_base64()
    return plot_url, img_buffer

def get_column_statistics(column):
    """获取列的统计信息"""
    col_info = column_info[column]
    data = processed_df[column].dropna()
    
    stats = {
        'column_name': column,
        'data_type': col_info['type'],
        'total_count': len(processed_df),
        'valid_count': len(data),
        'null_count': col_info['null_count'],
        'unique_count': col_info['unique_count'],
        'id': col_info['id']  # 添加列ID
    }
    
    if col_info['type'] in ['integer', 'float'] and len(data) > 0:
        stats.update({
            'mean': f"{data.mean():.2f}",
            'median': f"{data.median():.2f}",
            'std': f"{data.std():.2f}",
            'min': f"{data.min():.2f}",
            'max': f"{data.max():.2f}"
        })
    
    elif col_info['type'] in ['categorical', 'boolean'] and len(data) > 0:
        value_counts = data.value_counts()
        stats.update({
            'most_common': value_counts.index[0] if len(value_counts) > 0 else 'N/A',
            'most_common_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'distribution': dict(value_counts.head(5))
        })
    
    return stats

def load_and_analyze_excel(file_path):
    """加载并分析Excel或CSV文件"""
    global processed_df
    
    path = Path(file_path)
    if not path.exists():
        print(f"文件不存在: {path}")
        return False
    
    # 根据文件扩展名决定如何读取
    file_ext = path.suffix.lower()
    if file_ext == '.xlsx':
        df = pd.read_excel(path)
        print(f"成功加载Excel文件: {path.name}")
    elif file_ext == '.csv':
        # 尝试用不同编码和分隔符读取CSV
        try:
            df = pd.read_csv(path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(path, encoding='gbk')
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding='ISO-8859-1')
        except pd.errors.ParserError:
            # 尝试其他分隔符
            try:
                df = pd.read_csv(path, encoding='utf-8', sep=';')
            except:
                df = pd.read_csv(path, encoding='gbk', sep=';')
        
        print(f"成功加载CSV文件: {path.name}")
    else:
        print(f"不支持的文件类型: {file_ext}")
        return False
    print(f"原始数据形状: {df.shape}")
    
    # 分析数据结构
    analyze_data_structure(df)
    
    # 清理和处理数据
    processed_df = clean_and_process_data(df)
    
    print(f"处理后数据形状: {processed_df.shape}")
    print(f"成功识别 {len(processed_df)} 条有效数据")
    print("\n列信息:")
    for col, info in column_info.items():
        print(f"  {col}: {info['type']} (唯一值: {info['unique_count']}, 空值: {info['null_count']})")
    
    return True

def allowed_file(filename):
    """检查文件是否为允许的扩展名"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html', 
                         column_info=column_info, 
                         data_summary=data_summary)

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    global current_file
    
    # 检查是否有文件被上传
    if 'file' not in request.files:
        flash('没有选择文件')
        return redirect(request.url)
        
    file = request.files['file']
    
    # 如果用户没有选择文件，浏览器也会发送一个没有文件名的空部分
    if file.filename == '':
        flash('没有选择文件')
        return redirect(request.url)
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # 生成唯一文件名避免覆盖
        unique_filename = str(uuid.uuid4()) + '_' + filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # 记录当前文件路径
        current_file = file_path
        
        # 加载并分析上传的Excel文件
        success = load_and_analyze_excel(file_path)
        
        if success:
            return redirect(url_for('index'))
        else:
            flash('文件分析失败，请确保文件格式正确')
            return redirect(request.url)
    else:
        flash('不支持的文件类型，请上传.xlsx或.csv文件')
        return redirect(request.url)

@app.route('/chart/<int:column_id>/<chart_type>')
def show_chart(column_id, chart_type):
    """显示指定列和类型的图表"""
    # 获取所有自定义参数
    exclude_zeros = request.args.get('exclude_zeros', 'false').lower() == 'true'
    y_step_size = request.args.get('y_step_size')
    x_step_size = request.args.get('x_step_size')
    y_min = request.args.get('y_min')
    y_max = request.args.get('y_max')
    x_min = request.args.get('x_min')
    x_max = request.args.get('x_max')
    chart_color = request.args.get('chart_color')
    y_label = request.args.get('y_label')
    x_label = request.args.get('x_label')
    chart_title = request.args.get('chart_title')
    # 通过ID查找列名
    if column_id not in id_column_map:
        return f"列ID不存在: {column_id}", 404
    
    column = id_column_map[column_id]
    if column not in column_info:
        return f"列不存在: '{column}'", 404
    
    # 验证图表类型是否适合该列
    col_type = column_info[column]['type']
    
    # 文本类型不提供图表
    if col_type == 'text':
        return "文本类型列不提供图表视图", 400
    valid_charts = {
        'categorical': ['pie', 'bar'],
        'boolean': ['pie', 'bar'],
        'integer': ['histogram', 'line', 'bar'],
        'float': ['histogram', 'line'],
        'datetime': ['line']
    }
    
    if chart_type not in valid_charts.get(col_type, []):
        return f"图表类型 {chart_type} 不适用于 {col_type} 类型的数据", 400
    
    chart_data, _ = generate_chart(
        column, 
        chart_type, 
        exclude_zeros=exclude_zeros,
        y_step_size=y_step_size,
        x_step_size=x_step_size,
        y_min=y_min,
        y_max=y_max,
        x_min=x_min,
        x_max=x_max,
        chart_color=chart_color,
        y_label=y_label,
        x_label=x_label,
        chart_title=chart_title
    )
    stats = get_column_statistics(column)
    
    return render_template('chart.html', 
                         chart_data=chart_data,
                         column=column,
                         chart_type=chart_type,
                         stats=stats,
                         data_summary=data_summary,
                         exclude_zeros=exclude_zeros,
                         y_step_size=y_step_size,
                         x_step_size=x_step_size,
                         y_min=y_min,
                         y_max=y_max,
                         x_min=x_min,
                         x_max=x_max,
                         chart_color=chart_color,
                         y_label=y_label,
                         x_label=x_label,
                         chart_title=chart_title)

@app.route('/download_chart/<int:column_id>/<chart_type>')
def download_chart(column_id, chart_type):
    """下载图表为PNG文件"""
    # 获取所有自定义参数
    exclude_zeros = request.args.get('exclude_zeros', 'false').lower() == 'true'
    y_step_size = request.args.get('y_step_size')
    x_step_size = request.args.get('x_step_size')
    y_min = request.args.get('y_min')
    y_max = request.args.get('y_max')
    x_min = request.args.get('x_min')
    x_max = request.args.get('x_max')
    chart_color = request.args.get('chart_color')
    y_label = request.args.get('y_label')
    x_label = request.args.get('x_label')
    chart_title = request.args.get('chart_title')
    # 通过ID查找列名
    if column_id not in id_column_map:
        return f"列ID不存在: {column_id}", 404
    
    column = id_column_map[column_id]
    if column not in column_info:
        return f"列不存在: '{column}'", 404
    
    # 验证图表类型
    col_type = column_info[column]['type']
    valid_charts = {
        'categorical': ['pie', 'bar'],
        'boolean': ['pie', 'bar'],
        'integer': ['histogram', 'line', 'bar'],
        'float': ['histogram', 'line'],
        'datetime': ['line']
    }
    
    if chart_type not in valid_charts.get(col_type, []):
        return f"图表类型 {chart_type} 不适用于 {col_type} 类型的数据", 400
    
    # 生成图表并获取图像数据
    _, img_buffer = generate_chart(
        column, 
        chart_type, 
        exclude_zeros=exclude_zeros,
        y_step_size=y_step_size,
        x_step_size=x_step_size,
        y_min=y_min,
        y_max=y_max,
        x_min=x_min,
        x_max=x_max,
        chart_color=chart_color,
        y_label=y_label,
        x_label=x_label,
        chart_title=chart_title
    )
    img_buffer.seek(0)
    
    # 设置文件名
    filename = f"{column}_{chart_type}_chart.png"
    
    # 将图像数据作为文件发送
    return send_file(
        img_buffer,
        mimetype='image/png',
        as_attachment=True,
        download_name=filename
    )

@app.route('/column/<int:column_id>')
def column_detail(column_id):
    """显示列的详细信息"""
    # 通过ID查找列名
    if column_id not in id_column_map:
        return f"列ID不存在: {column_id}", 404
    
    column = id_column_map[column_id]
    col_type = column_info[column]['type']
    if col_type == 'text':
        return "文本类型列不提供详情视图", 400
    
    stats = get_column_statistics(column)
    
    # 推荐的图表类型
    chart_recommendations = {
        'categorical': [('pie', '饼图'), ('bar', '柱状图')],
        'boolean': [('pie', '饼图'), ('bar', '柱状图')],
        'integer': [('histogram', '直方图'), ('bar', '柱状图'), ('line', '折线图')],
        'float': [('histogram', '直方图'), ('line', '折线图')],
        'datetime': [('line', '折线图')]
    }
    
    recommended_charts = chart_recommendations.get(col_type, [])
    
    return render_template('column_detail.html',
                         column=column,
                         stats=stats,
                         recommended_charts=recommended_charts,
                         data_summary=data_summary)

@app.route('/lookup_data/<int:column_id>')
def lookup_data(column_id):
    """根据条件查找数据"""
    # 通过ID查找列名
    if column_id not in id_column_map:
        return jsonify({"error": f"列ID不存在: {column_id}"}), 404
    
    column = id_column_map[column_id]
    if column not in column_info:
        return jsonify({"error": f"列不存在: '{column}'"}), 404
    
    col_info = column_info[column]
    data_type = col_info['type']
    
    # 获取查询参数
    try:
        if data_type in ['integer', 'float']:
            condition = request.args.get('condition')
            filtered_df = None
            
            if condition == 'between':
                min_value = request.args.get('min_value')
                max_value = request.args.get('max_value')
                
                if not min_value or not max_value:
                    return jsonify({"error": "范围查询需要提供最小值和最大值"}), 400
                
                try:
                    min_value = float(min_value)
                    max_value = float(max_value)
                except ValueError:
                    return jsonify({"error": f"无法将值转换为数值：最小值='{min_value}'，最大值='{max_value}'"}), 400
                
                filtered_df = processed_df[(processed_df[column] >= min_value) & 
                                          (processed_df[column] <= max_value)]
            else:
                value = request.args.get('value')
                if not value:
                    return jsonify({"error": "需要提供查询值"}), 400
                
                try:
                    value = float(value)
                except ValueError:
                    return jsonify({"error": f"无法将值 '{value}' 转换为数值"}), 400
                
                if condition == 'eq':
                    filtered_df = processed_df[processed_df[column] == value]
                elif condition == 'gt':
                    filtered_df = processed_df[processed_df[column] > value]
                elif condition == 'lt':
                    filtered_df = processed_df[processed_df[column] < value]
                elif condition == 'gte':
                    filtered_df = processed_df[processed_df[column] >= value]
                elif condition == 'lte':
                    filtered_df = processed_df[processed_df[column] <= value]
                else:
                    return jsonify({"error": f"不支持的条件: {condition}"}), 400
                
        elif data_type in ['categorical', 'boolean']:
            category = request.args.get('category')
            if not category:
                return jsonify({"error": "需要提供类别值"}), 400
            
            filtered_df = processed_df[processed_df[column] == category]
        else:
            return jsonify({"error": f"不支持的数据类型: {data_type}"}), 400
        
        # 处理分页
        total_count = len(filtered_df)
        per_page = 100  # 每页显示100条记录
        
        # 获取页码，默认为第1页
        page = request.args.get('page', 1, type=int)
        if page < 1:
            page = 1
            
        # 计算总页数
        total_pages = (total_count + per_page - 1) // per_page  # 向上取整
        if page > total_pages and total_pages > 0:
            page = total_pages
        
        # 截取当前页数据
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_data = filtered_df.iloc[start_idx:end_idx].copy()
        
        # 准备结果说明
        if total_count > 0:
            result_note = f"共{total_count}条结果，当前显示第{page}页，共{total_pages}页"
        else:
            result_note = "没有找到匹配的数据"
            
        # 保留完整数据用于全部下载
        full_data = filtered_df
        
        # 转换为JSON格式前处理NaN值和格式保留
        # 先将NaN替换为None，这样在JSON序列化时会被转换为null
        json_data = []
        
        # 获取数据类型信息，用于保持格式一致性
        dtypes = page_data.dtypes.to_dict()
        
        for record in page_data.replace({np.nan: None}).to_dict('records'):
            # 确保所有值都是JSON可序列化的，并保持格式一致性
            clean_record = {}
            for key, value in record.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    clean_record[key] = None
                # 确保日期类型保持原始字符串格式
                elif pd.api.types.is_datetime64_any_dtype(dtypes.get(key)):
                    if value is not None:
                        # 如果是pandas Timestamp，转换为原始字符串格式
                        if isinstance(value, pd.Timestamp):
                            clean_record[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            clean_record[key] = str(value)
                    else:
                        clean_record[key] = None
                else:
                    clean_record[key] = value
            json_data.append(clean_record)
            
        # 全部数据的提取（用于支持下载完整数据）
        all_data = []
        if total_count > per_page:  # 只有当数据量大于每页显示量时才处理全量数据
            all_data_dtypes = full_data.dtypes.to_dict()
            for record in full_data.replace({np.nan: None}).to_dict('records'):
                clean_record = {}
                for key, value in record.items():
                    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                        clean_record[key] = None
                    elif pd.api.types.is_datetime64_any_dtype(all_data_dtypes.get(key)):
                        if value is not None:
                            if isinstance(value, pd.Timestamp):
                                clean_record[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                            else:
                                clean_record[key] = str(value)
                        else:
                            clean_record[key] = None
                    else:
                        clean_record[key] = value
                all_data.append(clean_record)
        else:
            all_data = json_data  # 如果数据量少，页面数据就是全部数据
        
        result = {
            "columns": page_data.columns.tolist(),
            "data": json_data,
            "all_data": all_data if len(all_data) <= 10000 else [],  # 限制返回的全量数据大小
            "total_count": total_count,
            "current_page": page,
            "total_pages": total_pages,
            "per_page": per_page,
            "note": result_note,
            "has_more_data": len(all_data) > 10000  # 标记是否需要单独请求全量数据
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"处理请求时出错: {str(e)}"}), 500


def main():
    print("启动智能数据分析系统...")
    print("请通过上传功能上传Excel文件进行分析")
    # 自动打开默认浏览器
    webbrowser.open('http://127.0.0.1:5000')
    # 在生产环境中关闭debug模式
    app.run(debug=False, host="127.0.0.1", port=5000)

if __name__ == "__main__":
    main()