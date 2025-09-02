from pathlib import Path
import random
import pandas as pd
from datetime import datetime, timedelta
import string

def generate_random_name():
    """生成随机姓名"""
    surnames = ['张', '李', '王', '刘', '陈', '杨', '黄', '赵', '周', '吴', '徐', '孙', '马', '朱', '胡', '林', '郭', '何', '高', '罗']
    given_names = ['芳', '娜', '秀英', '敏', '静', '丽', '强', '磊', '军', '洋', '勇', '艳', '杰', '娟', '涛', '明', '超', '秀兰', '霞']
    return random.choice(surnames) + random.choice(given_names)

def generate_random_phone():
    """生成随机手机号"""
    prefixes = ['136', '137', '138', '139', '150', '151', '152', '157', '158', '159', '130', '131', '132', '155', '156']
    return random.choice(prefixes) + ''.join([str(random.randint(0, 9)) for _ in range(8)])

def generate_random_id_card():
    """生成随机身份证号"""
    area_code = '350205' # 厦门海沧区
    birth_year = random.randint(1930, 2020)
    birth_month = random.randint(1, 12)
    birth_day = random.randint(1, 28)
    birth_date = f"{birth_year}{birth_month:02d}{birth_day:02d}"
    sequence = f"{random.randint(1, 999):03d}"
    
    # 简化的校验码生成
    check_codes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'X']
    check_code = random.choice(check_codes)
    
    return area_code + birth_date + sequence + check_code

def generate_random_bank_card():
    """生成随机银行卡号及开户行"""
    banks = [
        ('农业银行嵩屿支行', '6228480070'),
        ('工商银行海沧支行', '6222020200'),
        ('建设银行海沧支行', '6217000130'),
        ('中国银行海沧支行', '6013820100'),
        ('招商银行厦门分行', '6214830100')
    ]
    bank_name, prefix = random.choice(banks)
    card_number = prefix + ''.join([str(random.randint(0, 9)) for _ in range(10)])
    return f"{card_number}（{bank_name}）"

def generate_random_address():
    """生成随机地址"""
    communities = ['鳌冠社区', '嵩屿社区', '海沧社区', '新阳社区', '囷瑶社区', '钟山社区', '青礁社区', '温厝社区']
    community = random.choice(communities)
    
    household_num = random.randint(1, 999)
    residence_road = random.choice(['嵩屿路', '海沧大道', '钟林路', '新阳路', '囷瑶路'])
    residence_num = random.randint(1, 200)
    room_num = random.randint(101, 2999)
    
    return f"户籍：{community}{household_num}号\n居住：{residence_road}{residence_num}号{room_num}室"

def generate_random_birth_date():
    """生成随机出生年月"""
    start_date = datetime(1930, 1, 1)
    end_date = datetime(2020, 12, 31)
    time_between = end_date - start_date
    days_between = time_between.days
    random_days = random.randrange(days_between)
    birth_date = start_date + timedelta(days=random_days)
    return birth_date.strftime("%Y/%m/%d")

def generate_subsidy_info():
    """生成补贴信息"""
    subsidies = {}
    
    # 高龄老人补贴
    if random.choice([True, False]):
        subsidies['高龄老人补贴'] = '是'
        subsidies['高龄补贴金额'] = random.choice([200, 300, 500])
    else:
        subsidies['高龄老人补贴'] = '否'
        subsidies['高龄补贴金额'] = 0
    
    # 重残补贴
    if random.choice([True, False]):
        subsidies['重残补贴'] = '是'
        subsidies['重残补贴金额'] = random.choice([100, 150, 200])
    else:
        subsidies['重残补贴'] = '否'
        subsidies['重残补贴金额'] = 0
    
    # 困残补贴
    if random.choice([True, False]):
        subsidies['困残补贴'] = '是'
        subsidies['困残补贴金额'] = random.choice([50, 80, 100])
    else:
        subsidies['困残补贴'] = '否'
        subsidies['困残补贴金额'] = 0
    
    # 一户多残低保
    if random.choice([True, False]):
        subsidies['一户多残低保'] = '是'
        subsidies['一户多残金额'] = random.choice([300, 400, 500])
    else:
        subsidies['一户多残低保'] = '否'
        subsidies['一户多残金额'] = 0
    
    # 低保
    if random.choice([True, False]):
        subsidies['低保'] = '是'
        subsidies['低保金额'] = random.choice([800, 1000, 1200])
        subsidies['低保备注'] = random.choice(['无', '正常', '待审核'])
    else:
        subsidies['低保'] = '否'
        subsidies['低保金额'] = 0
        subsidies['低保备注'] = '不适用'
    
    # 低收入群体
    if random.choice([True, False]):
        subsidies['低收入群体'] = '是'
        subsidies['低收入备注'] = random.choice(['无', '正常'])
    else:
        subsidies['低收入群体'] = '否'
        subsidies['低收入备注'] = '不适用'
    
    # 事实无人抚养儿童
    if random.choice([True, False]):
        subsidies['事实无人抚养儿童'] = '是'
        subsidies['事无儿童补贴金额'] = random.choice([600, 800, 1000])
    else:
        subsidies['事实无人抚养儿童'] = '否'
        subsidies['事无儿童补贴金额'] = 0
    
    # 特困补贴
    if random.choice([True, False]):
        subsidies['特困补贴'] = '是'
        subsidies['特困补贴金额'] = random.choice([1500, 1800, 2000])
    else:
        subsidies['特困补贴'] = '否'
        subsidies['特困补贴金额'] = 0
    
    subsidies['备注'] = random.choice(['', '无', '正常', '待核实'])
    
    return subsidies

def generate_random_data(n=1000):
    """生成n条随机数据"""
    data = []
    communities = ['鳌冠社区', '嵩屿社区', '海沧社区', '新阳社区', '囷瑶社区', '钟山社区', '青礁社区', '温厝社区']
    
    for i in range(1, n + 1):
        subsidy_info = generate_subsidy_info()
        
        row = {
            '序号': i,
            '村居': random.choice(communities),
            '姓名': generate_random_name(),
            '性别': random.choice(['男', '女']),
            '出生年月': generate_random_birth_date(),
            '联系电话': generate_random_phone(),
            '身份证号': generate_random_id_card(),
            '银行卡号及开户行': generate_random_bank_card(),
            '地址（户籍、居住）': generate_random_address(),
            '高龄老人补贴（是/否）': subsidy_info['高龄老人补贴'],
            '高龄补贴金额': subsidy_info['高龄补贴金额'],
            '重残补贴（是/否）': subsidy_info['重残补贴'],
            '重残补贴金额': subsidy_info['重残补贴金额'],
            '困残补贴（是/否）': subsidy_info['困残补贴'],
            '困残补贴金额': subsidy_info['困残补贴金额'],
            '一户多残低保（是/否）': subsidy_info['一户多残低保'],
            '一户多残金额': subsidy_info['一户多残金额'],
            '备注': subsidy_info['备注'],
            '低保（是/否）': subsidy_info['低保'],
            '低保金额': subsidy_info['低保金额'],
            '低保备注': subsidy_info['低保备注'],
            '低收入群体（是/否）': subsidy_info['低收入群体'],
            '低收入备注': subsidy_info['低收入备注'],
            '事实无人抚养儿童（是/否）': subsidy_info['事实无人抚养儿童'],
            '事无儿童补贴金额': subsidy_info['事无儿童补贴金额'],
            '特困补贴（是/否）': subsidy_info['特困补贴'],
            '特困补贴金额': subsidy_info['特困补贴金额']
        }
        data.append(row)
    
    return data

def save_to_excel(data, file_path):
    """保存数据到指定Excel文件"""
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False, engine='openpyxl')
    
    print(f"数据已保存到: {output_path.absolute()}")
    return output_path

# 主程序
if __name__ == "__main__":
    # 生成1000条数据（可以修改数量）
    n = 1000
    print(f"开始生成{n}条随机数据...")
    
    # 生成数据
    random_data = generate_random_data(n)
    
    # 保存到指定路径
    output_file_path = "/Users/xyy/git_syn/qxy_app/demo/gen_data_v1.xlsx"
    save_to_excel(random_data, output_file_path)
    
    print(f"总共生成了{len(random_data)}条数据")