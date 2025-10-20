import random
from PIL import Image, ImageDraw
import io
import torch
import gc

class SyntheticDataGenerator:
    def __init__(self):
        self.chart_types = ['bar', 'line', 'pie', 'scatter', 'area', 'histogram']
        self.categories = ['Sales', 'Revenue', 'Profit', 'Growth', 'Market Share', 'Performance']
        self.time_periods = ['2020', '2021', '2022', '2023', '2024', 'Q1', 'Q2', 'Q3', 'Q4']
        
        # GPU内存优化
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_clean_interval = 100  # 每100个样本清理一次内存
    
    def generate_sample(self, idx):
        """生成一个合成数据样本（优化内存使用）"""
        chart_type = random.choice(self.chart_types)
        category = random.choice(self.categories)
        time_period = random.choice(self.time_periods)
        
        # 根据图表类型创建不同的图像和问题
        if chart_type == 'bar':
            img, question, answer = self._create_bar_chart_data(category, time_period, idx)
        elif chart_type == 'line':
            img, question, answer = self._create_line_chart_data(category, time_period, idx)
        elif chart_type == 'pie':
            img, question, answer = self._create_pie_chart_data(category, time_period, idx)
        elif chart_type == 'scatter':
            img, question, answer = self._create_scatter_plot_data(category, time_period, idx)
        elif chart_type == 'area':
            img, question, answer = self._create_area_chart_data(category, time_period, idx)
        else:  # histogram
            img, question, answer = self._create_histogram_data(category, time_period, idx)
        
        # 定期清理GPU内存
        if idx % self.memory_clean_interval == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        return {
            'image': img,
            'question': question,
            'answer': [answer],
            'source': f'synthetic_{chart_type}',
            'ocr_text': f"{chart_type.capitalize()} chart showing {category} for {time_period}"
        }
    
    def _create_bar_chart_data(self, category, time_period, idx):
        """创建柱状图数据"""
        values = [random.randint(10, 100) for _ in range(5)]
        labels = ['A', 'B', 'C', 'D', 'E']
        
        # 创建柱状图图像
        img = self._create_bar_chart_image(labels, values, f"{category} {time_period}")
        
        # 生成问题和答案
        question_types = [
            f"What is the value of {labels[2]} in this bar chart showing {category} for {time_period}?",
            f"Which category has the highest value in this {category} bar chart for {time_period}?",
            f"What is the total value shown in this {category} bar chart for {time_period}?"
        ]
        
        question = random.choice(question_types)
        
        if "value of" in question:
            answer = str(values[2])
        elif "highest" in question:
            max_idx = values.index(max(values))
            answer = labels[max_idx]
        else:
            answer = str(sum(values))
        
        return img, question, answer
    
    def _create_line_chart_data(self, category, time_period, idx):
        """创建折线图数据"""
        values = [random.randint(20, 80) for _ in range(6)]
        labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        
        # 创建折线图图像
        img = self._create_line_chart_image(labels, values, f"{category} {time_period}")
        
        # 生成问题和答案
        question_types = [
            f"What was the value in {labels[3]} for this {category} line chart in {time_period}?",
            f"What is the trend shown in this {category} line chart for {time_period}?",
            f"What is the difference between the highest and lowest values in this {category} line chart?"
        ]
        
        question = random.choice(question_types)
        
        if labels[3] in question:
            answer = str(values[3])
        elif "trend" in question:
            if values[-1] > values[0]:
                answer = "increasing"
            else:
                answer = "decreasing"
        else:
            answer = str(max(values) - min(values))
        
        return img, question, answer
    
    def _create_pie_chart_data(self, category, time_period, idx):
        """创建饼图数据"""
        values = [random.randint(10, 40) for _ in range(4)]
        labels = ['Product A', 'Product B', 'Product C', 'Product D']
        
        # 创建饼图图像
        img = self._create_pie_chart_image(labels, values, f"{category} {time_period}")
        
        # 生成问题和答案
        question_types = [
            f"What percentage does {labels[1]} represent in this {category} pie chart for {time_period}?",
            f"Which product has the largest share in this {category} pie chart for {time_period}?",
            f"What is the total value represented in this {category} pie chart for {time_period}?"
        ]
        
        question = random.choice(question_types)
        
        if "percentage" in question:
            total = sum(values)
            percentage = round((values[1] / total) * 100)
            answer = f"{percentage}%"
        elif "largest" in question:
            max_idx = values.index(max(values))
            answer = labels[max_idx]
        else:
            answer = str(sum(values))
        
        return img, question, answer
    
    def _create_scatter_plot_data(self, category, time_period, idx):
        """创建散点图数据"""
        x_values = [random.randint(1, 100) for _ in range(20)]
        y_values = [random.randint(1, 100) for _ in range(20)]
        
        # 创建散点图图像
        img = self._create_scatter_plot_image(x_values, y_values, f"{category} {time_period}")
        
        # 生成问题和答案
        question_types = [
            f"How many data points are shown in this {category} scatter plot for {time_period}?",
            f"What is the range of x-values in this {category} scatter plot for {time_period}?",
            f"Are there any obvious clusters in this {category} scatter plot for {time_period}?"
        ]
        
        question = random.choice(question_types)
        
        if "data points" in question:
            answer = "20"
        elif "range" in question:
            answer = f"{min(x_values)}-{max(x_values)}"
        else:
            answer = "no obvious clusters"
        
        return img, question, answer
    
    def _create_area_chart_data(self, category, time_period, idx):
        """创建面积图数据"""
        values = [random.randint(30, 90) for _ in range(5)]
        labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
        
        # 创建面积图图像
        img = self._create_area_chart_image(labels, values, f"{category} {time_period}")
        
        # 生成问题和答案
        question_types = [
            f"What is the area under the curve for {labels[2]} in this {category} area chart for {time_period}?",
            f"What is the cumulative value shown in this {category} area chart for {time_period}?",
            f"Which quarter shows the highest value in this {category} area chart for {time_period}?"
        ]
        
        question = random.choice(question_types)
        
        if "area" in question and "curve" in question:
            answer = str(values[2])
        elif "cumulative" in question:
            answer = str(sum(values))
        else:
            max_idx = values.index(max(values))
            answer = labels[max_idx]
        
        return img, question, answer
    
    def _create_histogram_data(self, category, time_period, idx):
        """创建直方图数据"""
        values = [random.randint(5, 25) for _ in range(6)]
        bins = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60']
        
        # 创建直方图图像
        img = self._create_histogram_image(bins, values, f"{category} {time_period}")
        
        # 生成问题和答案
        question_types = [
            f"How many items fall in the {bins[2]} range in this {category} histogram for {time_period}?",
            f"What is the most frequent range in this {category} histogram for {time_period}?",
            f"What is the total number of items represented in this {category} histogram for {time_period}?"
        ]
        
        question = random.choice(question_types)
        
        if bins[2] in question:
            answer = str(values[2])
        elif "most frequent" in question:
            max_idx = values.index(max(values))
            answer = bins[max_idx]
        else:
            answer = str(sum(values))
        
        return img, question, answer
    
    # 图表图像创建方法（保持原有逻辑，优化内存使用）
    def _create_bar_chart_image(self, labels, values, title):
        """创建柱状图图像（优化内存使用）"""
        img = Image.new('RGB', (300, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # 绘制坐标轴
        draw.line([(50, 50), (50, 150)], fill='black', width=2)
        draw.line([(50, 150), (250, 150)], fill='black', width=2)
        
        # 绘制柱状图
        bar_width = 30
        max_value = max(values) if max(values) > 0 else 1  # 避免除零错误
        
        for i, (label, value) in enumerate(zip(labels, values)):
            x = 70 + i * 40
            height = int((value / max_value) * 80)
            
            # 确保高度至少为1像素，避免y1小于y0
            if height == 0:
                height = 1
            
            # 绘制柱子（确保y1 > y0）
            y_top = 150 - height
            y_bottom = 150
            draw.rectangle([x, y_top, x + bar_width, y_bottom], fill='blue')
            
            # 添加标签
            draw.text((x + 5, 155), label, fill='black')
            draw.text((x + 5, y_top - 10), str(value), fill='black')
        
        # 添加标题
        draw.text((100, 20), title, fill='black')
        
        return self._image_to_bytes(img)
    
    def _create_line_chart_image(self, labels, values, title):
        """创建折线图图像（优化内存使用）"""
        img = Image.new('RGB', (300, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # 绘制坐标轴
        draw.line([(50, 50), (50, 150)], fill='black', width=2)
        draw.line([(50, 150), (250, 150)], fill='black', width=2)
        
        # 绘制折线
        max_value = max(values)
        points = []
        
        for i, (label, value) in enumerate(zip(labels, values)):
            x = 70 + i * 35
            y = 150 - int((value / max_value) * 80)
            points.append((x, y))
            
            # 添加数据点
            draw.ellipse([x-2, y-2, x+2, y+2], fill='red')
            draw.text((x-5, y+5), str(value), fill='black')
            draw.text((x-5, 155), label, fill='black')
        
        # 连接数据点
        if len(points) > 1:
            for i in range(len(points)-1):
                draw.line([points[i], points[i+1]], fill='red', width=2)
        
        # 添加标题
        draw.text((100, 20), title, fill='black')
        
        return self._image_to_bytes(img)
    
    def _create_pie_chart_image(self, labels, values, title):
        """创建饼图图像（优化内存使用）"""
        img = Image.new('RGB', (300, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # 绘制饼图
        total = sum(values)
        start_angle = 0
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        
        center_x, center_y = 150, 100
        radius = 60
        
        for i, value in enumerate(values):
            angle = 360 * value / total
            # 绘制扇形
            draw.pieslice([center_x-radius, center_y-radius, center_x+radius, center_y+radius], 
                         start_angle, start_angle + angle, fill=colors[i % len(colors)])
            start_angle += angle
        
        # 添加标题
        draw.text((100, 20), title, fill='black')
        
        return self._image_to_bytes(img)
    
    def _create_scatter_plot_image(self, x_values, y_values, title):
        """创建散点图图像（优化内存使用）"""
        img = Image.new('RGB', (300, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # 绘制坐标轴
        draw.line([(50, 50), (50, 150)], fill='black', width=2)
        draw.line([(50, 150), (250, 150)], fill='black', width=2)
        
        # 绘制散点
        max_x, max_y = max(x_values), max(y_values)
        
        for x, y in zip(x_values, y_values):
            plot_x = 50 + int((x / max_x) * 180)
            plot_y = 150 - int((y / max_y) * 80)
            draw.ellipse([plot_x-2, plot_y-2, plot_x+2, plot_y+2], fill='blue')
        
        # 添加标题
        draw.text((100, 20), title, fill='black')
        
        return self._image_to_bytes(img)
    
    def _create_area_chart_image(self, labels, values, title):
        """创建面积图图像（优化内存使用）"""
        img = Image.new('RGB', (300, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # 绘制坐标轴
        draw.line([(50, 50), (50, 150)], fill='black', width=2)
        draw.line([(50, 150), (250, 150)], fill='black', width=2)
        
        # 绘制面积图
        max_value = max(values)
        points = [(50, 150)]
        
        for i, (label, value) in enumerate(zip(labels, values)):
            x = 70 + i * 40
            y = 150 - int((value / max_value) * 80)
            points.append((x, y))
            
            draw.text((x-5, 155), label, fill='black')
        
        points.append((250, 150))
        
        # 填充面积
        draw.polygon(points, fill='lightblue')
        
        # 绘制折线
        for i in range(1, len(points)-1):
            draw.line([points[i], points[i+1]], fill='blue', width=2)
        
        # 添加标题
        draw.text((100, 20), title, fill='black')
        
        return self._image_to_bytes(img)
    
    def _create_histogram_image(self, bins, values, title):
        """创建直方图图像（优化内存使用）"""
        img = Image.new('RGB', (300, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # 绘制坐标轴
        draw.line([(50, 50), (50, 150)], fill='black', width=2)
        draw.line([(50, 150), (250, 150)], fill='black', width=2)
        
        # 绘制直方图
        max_value = max(values) if max(values) > 0 else 1  # 避免除零错误
        bar_width = 30
        
        for i, (bin_label, value) in enumerate(zip(bins, values)):
            x = 70 + i * 35
            height = int((value / max_value) * 80)
            
            # 确保高度至少为1像素，避免y1小于y0
            if height == 0:
                height = 1
            
            # 绘制柱子（确保y1 > y0）
            y_top = 150 - height
            y_bottom = 150
            draw.rectangle([x, y_top, x + bar_width, y_bottom], fill='green')
            
            # 添加标签
            draw.text((x, 155), bin_label, fill='black')
            draw.text((x, y_top - 10), str(value), fill='black')
        
        # 添加标题
        draw.text((100, 20), title, fill='black')
        
        return self._image_to_bytes(img)
    
    def _image_to_bytes(self, img):
        """将PIL图像转换为字节（优化内存使用）"""
        try:
            buf = io.BytesIO()
            img.save(buf, format='PNG', optimize=True)  # 使用优化选项
            buf.seek(0)
            return {'bytes': buf.getvalue()}
        except Exception as e:
            print(f"图像转换失败: {e}")
            # 返回空白图像
            blank_img = Image.new('RGB', (224, 224), color='white')
            blank_buf = io.BytesIO()
            blank_img.save(blank_buf, format='PNG')
            blank_buf.seek(0)
            return {'bytes': blank_buf.getvalue()}