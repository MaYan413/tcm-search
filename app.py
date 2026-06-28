#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中医药辅助诊疗系统 - 多轮预问诊后端
Flask API Server + 静态文件服务
"""

import json
import os
import re
import uuid
import logging
from collections import Counter, defaultdict
from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS

# ============================================================
# 配置
# ============================================================
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'tcm-consultation-secret-key-2024')
CORS(app, supports_credentials=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据目录
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
NUM_DATA_FILES = 16

# LLM 配置（OpenAI 兼容接口）
LLM_API_KEY = os.environ.get('LLM_API_KEY', '')
LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.deepseek.com')
LLM_MODEL = os.environ.get('LLM_MODEL', 'deepseek-chat')
USE_LLM = bool(LLM_API_KEY)

# 会话状态存储（生产环境应使用 Redis）
session_store = {}

# ============================================================
# 数据加载（按需加载，避免免费版 Render 512MB 内存溢出）
# ============================================================
all_records = []
_data_loaded = False
_data_load_failed = False

def load_all_data():
    """加载全部16个JSON数据文件到内存（仅在需要时调用）"""
    global all_records, _data_loaded, _data_load_failed
    if _data_loaded:
        return
    if _data_load_failed:
        return
    all_records = []
    loaded_count = 0
    for i in range(1, NUM_DATA_FILES + 1):
        filepath = os.path.join(DATA_DIR, f'medical_records_part{i}.json')
        if not os.path.exists(filepath):
            logger.warning(f'数据文件不存在: {filepath}')
            continue
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                records = json.load(f)
                # 只保留必要字段以节省内存
                slim_records = []
                for r in records:
                    slim_records.append({
                        'symptoms': r.get('symptoms', '') or '',
                        'diagnosis': r.get('diagnosis', '') or '',
                        'formula': r.get('formula', '') or '',
                        'herbs': r.get('herbs', '') or '',
                    })
                all_records.extend(slim_records)
                loaded_count += len(slim_records)
            logger.info(f'加载 {os.path.basename(filepath)}: {len(records)} 条')
        except MemoryError:
            logger.error(f'内存不足，停止加载。已加载 {loaded_count} 条')
            _data_load_failed = True
            all_records = []
            break
        except Exception as e:
            logger.error(f'加载 {filepath} 失败: {e}')
    logger.info(f'总计加载 {len(all_records)} 条医案记录')
    _data_loaded = True


def parse_symptoms_text(text):
    """从症状文本中提取结构化信息"""
    if not text:
        return {}
    info = {
        'age': None,
        'gender': None,
        'chief_complaint': '',
        'symptoms': [],
        'tongue': '',
        'pulse': '',
        'diagnosis_in_text': '',
    }
    # 提取年龄
    age_match = re.search(r'年龄[:：]\s*(\d+)', text)
    if age_match:
        info['age'] = int(age_match.group(1))
    # 提取性别
    gender_match = re.search(r'性别[:：]\s*([男女])', text)
    if gender_match:
        info['gender'] = gender_match.group(1)
    # 提取主诉
    cc_match = re.search(r'主诉[:：]\s*([^现]*)', text)
    if cc_match:
        info['chief_complaint'] = cc_match.group(1).strip()
    # 提取舌诊
    tongue_match = re.search(r'舌[色质苔象态][^，,。；;]*(?:[，,][^，,。；;]*)?', text)
    if tongue_match:
        info['tongue'] = tongue_match.group(0).strip()
    # 提取脉诊
    pulse_match = re.search(r'脉[象搏][^，,。；;]*(?:[，,][^，,。；;]*)?', text)
    if pulse_match:
        info['pulse'] = pulse_match.group(0).strip()
    # 提取证型
    diag_match = re.search(r'证型[:：]\s*([^，,。；;中药治法]*)', text)
    if diag_match:
        info['diagnosis_in_text'] = diag_match.group(1).strip()
    # 提取症状关键词
    symptom_keywords = re.findall(r'([一-鿿]{2,6}(?:\([^)]*\))?)', text)
    info['symptoms'] = [s for s in symptom_keywords if len(s) >= 2 and not s.startswith(('年龄', '性别', '身高', '体重', '主诉', '现病史', '体格', '检查'))]
    return info


def search_by_symptom(keyword, gender=None, age=None, additional_filters=None):
    """
    按症状关键词搜索医案
    返回匹配的记录列表
    """
    keyword_lower = keyword.lower().strip()
    results = []
    for record in all_records:
        symptoms_text = (record.get('symptoms', '') or '').lower()
        diagnosis_text = (record.get('diagnosis', '') or '').lower()
        # 关键词匹配
        if keyword_lower in symptoms_text or keyword_lower in diagnosis_text:
            # 性别过滤
            if gender:
                parsed = parse_symptoms_text(record.get('symptoms', ''))
                if parsed.get('gender') and parsed['gender'] != gender:
                    continue
            # 年龄过滤（±15岁）
            if age is not None:
                parsed = parse_symptoms_text(record.get('symptoms', ''))
                record_age = parsed.get('age')
                if record_age is not None and abs(record_age - age) > 15:
                    continue
            # 额外过滤条件
            if additional_filters:
                match = True
                for filter_key, filter_vals in additional_filters.items():
                    if isinstance(filter_vals, list):
                        if not any(v.lower() in symptoms_text for v in filter_vals):
                            match = False
                            break
                    elif isinstance(filter_vals, str):
                        if filter_vals.lower() not in symptoms_text:
                            match = False
                            break
                if not match:
                    continue
            results.append(record)
    return results


def extract_symptom_features(records):
    """从匹配记录中提取症状特征分布"""
    symptom_counter = Counter()
    diagnosis_counter = Counter()
    formula_counter = Counter()
    herb_counter = Counter()
    tongue_set = set()
    pulse_set = set()
    for r in records:
        parsed = parse_symptoms_text(r.get('symptoms', ''))
        for s in parsed.get('symptoms', []):
            if len(s) >= 2:
                symptom_counter[s] += 1
        diag = (r.get('diagnosis', '') or '').strip()
        if diag and '证型' in diag:
            diag_clean = re.sub(r'^证型[:：]\s*', '', diag)
            if diag_clean:
                diagnosis_counter[diag_clean] += 1
        formula = (r.get('formula', '') or '').strip()
        if formula:
            # 取第一个方剂名
            first_formula = formula.split()[0] if formula.split() else formula[:20]
            formula_counter[first_formula] += 1
        herbs = (r.get('herbs', '') or '').strip()
        if herbs:
            for h in herbs.split():
                h_clean = re.sub(r'[\d.]+$', '', h).strip()
                if h_clean and len(h_clean) >= 2:
                    herb_counter[h_clean] += 1
        if parsed.get('tongue'):
            tongue_set.add(parsed['tongue'])
        if parsed.get('pulse'):
            pulse_set.add(parsed['pulse'])
    return {
        'common_symptoms': symptom_counter.most_common(30),
        'common_diagnoses': diagnosis_counter.most_common(10),
        'common_formulas': formula_counter.most_common(10),
        'common_herbs': herb_counter.most_common(20),
        'tongue_patterns': list(tongue_set),
        'pulse_patterns': list(pulse_set),
    }


# ============================================================
# 追问问题模板（按症状类别）
# ============================================================
FOLLOWUP_QUESTION_BANK = {
    # 疼痛类
    "疼痛|痛": [
        {"question": "疼痛的具体部位在哪里？", "options": [], "multi_select": False, "dynamic_options": True, "category": "部位"},
        {"question": "疼痛的性质是怎样的？", "options": ["胀痛", "刺痛/针扎样", "隐痛/绵绵不休", "灼痛/烧灼感", "冷痛", "绞痛", "游走性疼痛", "固定不移"], "multi_select": False, "category": "性质"},
        {"question": "疼痛在什么情况下会加重？", "options": ["遇冷加重", "遇热加重", "劳累后加重", "夜间加重", "按压后加重", "情绪波动后加重", "饥饿时加重", "饭后加重"], "multi_select": True, "category": "诱因"},
        {"question": "疼痛在什么情况下会缓解？", "options": ["休息后缓解", "得温缓解", "得凉缓解", "按压后缓解", "进食后缓解", "活动后缓解"], "multi_select": True, "category": "缓解因素"},
    ],
    # 咳嗽/呼吸类
    "咳嗽|气喘|哮喘|痰": [
        {"question": "咳嗽有痰吗？痰的颜色和质地如何？", "options": ["干咳无痰", "白痰/清稀", "黄痰/黏稠", "痰中带血丝", "泡沫样痰", "痰多易咳", "痰少难咳"], "multi_select": False, "category": "痰"},
        {"question": "咳嗽在什么时间最明显？", "options": ["晨起", "夜间", "午后", "全天", "遇冷空气时", "平卧时"], "multi_select": True, "category": "时间规律"},
        {"question": "有无以下伴随症状？", "options": ["发热", "恶寒怕冷", "咽喉痒痛", "胸闷", "气短", "鼻塞流涕", "自汗", "盗汗"], "multi_select": True, "category": "伴随"},
    ],
    # 消化类
    "胃|腹|消化|呕吐|恶心|腹泻|便秘|便溏|食欲": [
        {"question": "不适症状与饮食有什么关系？", "options": ["饭前明显", "饭后加重", "饥饿时加重", "与饮食无关"], "multi_select": False, "category": "饮食关系"},
        {"question": "具体是哪种不适感觉？", "options": ["胀满感", "隐痛", "灼热感/烧心", "反酸", "嗳气/打嗝", "恶心", "呕吐", "嘈杂（说不清的难受）"], "multi_select": True, "category": "感觉类型"},
        {"question": "大便情况如何？", "options": ["大便正常", "便秘/便干", "便溏/不成形", "腹泻/水样便", "大便黏腻不爽", "时干时稀", "便血/黑便"], "multi_select": False, "category": "大便"},
        {"question": "有无以下伴随症状？", "options": ["口苦", "口臭", "口干口渴", "口淡无味", "腹胀", "肠鸣", "食欲不振", "善饥易饿"], "multi_select": True, "category": "伴随"},
    ],
    # 睡眠/精神类
    "失眠|睡眠|入睡|多梦|易醒|焦虑|抑郁|烦躁": [
        {"question": "睡眠问题的具体表现是什么？", "options": ["入睡困难", "睡后易醒", "多梦纷扰", "早醒（凌晨醒来无法再睡）", "彻夜不眠", "睡眠浅/似睡非睡"], "multi_select": True, "category": "表现"},
        {"question": "什么情况下会加重失眠？", "options": ["思虑过度/压力大", "饮食不节（过饱/咖啡茶等）", "环境改变", "身体不适（疼痛/咳嗽等）", "无明显诱因"], "multi_select": False, "category": "诱因"},
        {"question": "白天精神状态如何？", "options": ["精神尚可", "疲倦乏力", "头晕头昏", "心烦易怒", "注意力不集中", "心悸心慌"], "multi_select": True, "category": "日间状态"},
    ],
    # 头面类
    "头晕|头昏|眩晕|耳鸣|头痛": [
        {"question": "头晕/眩晕的性质是怎样的？", "options": ["天旋地转（旋转性）", "头重脚轻/如踩棉花", "眼前发黑", "昏沉不清/如裹湿布", "站立不稳"], "multi_select": False, "category": "性质"},
        {"question": "头晕发作与什么有关？", "options": ["体位改变时（如起床/转头）", "劳累后", "饥饿时", "情绪波动后", "无明显诱因", "持续性"], "multi_select": True, "category": "诱因"},
        {"question": "有无以下伴随症状？", "options": ["恶心呕吐", "耳鸣", "视物模糊", "心慌", "出汗", "手抖", "面色苍白"], "multi_select": True, "category": "伴随"},
    ],
    # 妇科类
    "月经|经期|痛经|带下|白带": [
        {"question": "月经周期如何？", "options": ["周期正常（28±7天）", "月经先期（提前>7天）", "月经后期（推后>7天）", "月经先后无定期", "闭经"], "multi_select": False, "category": "周期"},
        {"question": "月经量如何？", "options": ["量中", "量多", "量少", "淋漓不尽", "量时多时少"], "multi_select": False, "category": "经量"},
        {"question": "经色和质地如何？", "options": ["色淡红/质稀", "色鲜红/质稠", "色暗红/有血块", "色紫黑/血块多"], "multi_select": False, "category": "经色"},
        {"question": "有无以下伴随症状？", "options": ["经行腹痛", "腰酸", "乳房胀痛", "情绪波动", "头痛", "浮肿", "发热"], "multi_select": True, "category": "伴随"},
    ],
    # 泌尿类
    "小便|尿|水肿|浮肿": [
        {"question": "小便情况如何？", "options": ["小便正常", "小便频繁", "小便短少/不畅", "小便涩痛", "夜尿增多", "小便清长", "小便黄赤"], "multi_select": False, "category": "小便"},
        {"question": "有无以下伴随症状？", "options": ["腰酸腰痛", "水肿/浮肿", "口干口渴", "怕冷", "发热", "乏力"], "multi_select": True, "category": "伴随"},
    ],
    # 寒热/出汗类
    "发热|怕冷|恶寒|出汗|盗汗|自汗|潮热": [
        {"question": "寒热感觉是怎样的？", "options": ["恶寒发热（怕冷同时发热）", "但热不寒（只发热不怕冷）", "但寒不热（只怕冷不发热）", "寒热往来（忽冷忽热）", "潮热（定时发热如潮水）", "五心烦热（手心脚心胸口发热）"], "multi_select": False, "category": "寒热"},
        {"question": "出汗情况如何？", "options": ["自汗（白天不活动也出汗）", "盗汗（睡时出汗醒后汗止）", "无汗", "汗出过多", "局部出汗（头汗/手足汗等）"], "multi_select": False, "category": "汗出"},
    ],
    # 全身性
    "乏力|疲倦|疲劳|虚弱|消瘦|肥胖": [
        {"question": "乏力疲倦的具体表现？", "options": ["全身乏力/少气懒言", "四肢沉重", "活动后加重", "休息后仍不缓解", "午后尤甚"], "multi_select": True, "category": "表现"},
        {"question": "有无以下伴随症状？", "options": ["气短（说话无力/上气不接下气）", "心慌心悸", "头晕", "食欲不振", "睡眠不佳", "面色萎黄或苍白"], "multi_select": True, "category": "伴随"},
    ],
    # 皮肤类
    "皮肤|湿疹|皮疹|瘙痒|痤疮|疮": [
        {"question": "皮损/皮疹的形态是什么？", "options": ["红斑", "丘疹/小疙瘩", "水疱", "脓疱", "脱屑", "结痂", "风团/荨麻疹样"], "multi_select": True, "category": "形态"},
        {"question": "瘙痒的程度和特点？", "options": ["轻度瘙痒", "剧烈瘙痒", "夜间加重", "遇热加重", "遇冷加重", "无明显瘙痒"], "multi_select": False, "category": "瘙痒"},
    ],
}

# 通用追问模板（当无法匹配特定类别时）
GENERIC_FOLLOWUP_QUESTIONS = [
    {"question": "这个症状持续多长时间了？", "options": ["3天以内", "3-7天", "1-2周", "2周-1个月", "1个月以上", "反复发作/迁延不愈"], "multi_select": False, "category": "病程"},
    {"question": "症状的严重程度如何？", "options": ["轻微/偶尔", "中度/影响日常", "重度/难以忍受"], "multi_select": False, "category": "程度"},
    {"question": "以前有过类似症状吗？", "options": ["首次出现", "偶尔发作", "反复发作", "持续存在"], "multi_select": False, "category": "病史"},
    {"question": "有无以下全身伴随症状？", "options": ["怕冷", "发热", "出汗异常", "口干口渴", "口苦", "疲倦乏力", "头晕", "心慌"], "multi_select": True, "category": "全身伴随"},
    {"question": "大便情况如何？", "options": ["正常", "便秘", "便溏/不成形", "腹泻", "黏腻不爽", "干稀不调"], "multi_select": False, "category": "大便"},
    {"question": "睡眠情况如何？", "options": ["睡眠正常", "入睡困难", "多梦", "易醒", "早醒", "嗜睡"], "multi_select": True, "category": "睡眠"},
    {"question": "舌象如何？（可对照镜子观察）", "options": ["舌淡", "舌红", "舌暗/紫", "舌胖/有齿痕", "舌瘦薄", "舌苔白", "舌苔黄", "舌苔厚/腻", "舌苔少/剥落", "不清楚"], "multi_select": True, "category": "舌象"},
]


def get_followup_questions_for_complaint(chief_complaint):
    """根据主诉获取匹配的追问问题模板"""
    matched_questions = []
    for pattern, questions in FOLLOWUP_QUESTION_BANK.items():
        if re.search(pattern, chief_complaint):
            matched_questions.extend(questions)
    if not matched_questions:
        matched_questions = GENERIC_FOLLOWUP_QUESTIONS[:]
    return matched_questions


def generate_dynamic_options(question_template, records, chief_complaint):
    """根据匹配的病例动态生成选项"""
    options = list(question_template.get('options', []))
    if not question_template.get('dynamic_options', False):
        return options
    # 从病例中提取相关症状作为选项
    features = extract_symptom_features(records)
    category = question_template.get('category', '')
    # 部位类：从症状中提取可能的部位
    body_parts = {
        '头部': ['头', '巅顶', '前额', '太阳穴', '后脑', '枕部'],
        '胸部': ['胸', '心前区', '胁肋', '乳房'],
        '腹部': ['上腹', '下腹', '脐周', '小腹', '少腹', '胃脘'],
        '四肢': ['肩', '肘', '腕', '手指', '膝', '踝', '足', '下肢', '上肢'],
        '腰部': ['腰', '背', '颈'],
    }
    if category == '部位':
        dynamic_opts = set()
        symptoms_text = ' '.join([r.get('symptoms', '') for r in records[:50]])
        for region, parts in body_parts.items():
            for part in parts:
                if part in symptoms_text:
                    dynamic_opts.add(f"{part}")
        options.extend(sorted(dynamic_opts)[:8])
    if not options:
        options = question_template.get('options', [])
    return options


# ============================================================
# TCM 术语解释词典（规则引擎后备）
# ============================================================
TCM_TERM_DICT = {
    "风寒袭络证": "外感风寒之邪侵袭经络，导致经络气血运行不畅。常见症状包括头痛、颈项强痛、恶寒发热、肢体酸痛等。治疗以祛风散寒、通络止痛为主。",
    "肝气郁结": "由于情志不畅、精神压力等因素导致肝的疏泄功能失调，气机郁滞。常见症状包括胸胁胀痛、情绪抑郁、善叹息、月经不调等。治疗以疏肝理气为主。",
    "肝郁脾虚证": "肝气郁结日久影响脾胃运化功能所致。既有肝郁的胸胁胀痛、情绪不畅，又有脾虚的食欲不振、腹胀便溏、疲倦乏力。治疗需疏肝健脾并重。",
    "脾胃虚弱": "脾胃的运化吸收功能减弱。常见症状包括食欲不振、饭后腹胀、大便稀溏、面色萎黄、疲倦乏力等。治疗以健脾益气为主。",
    "气血两虚": "气虚和血虚同时存在。常见症状包括面色苍白或萎黄、头晕目眩、心慌气短、疲倦乏力、失眠多梦等。治疗以补气养血为主。",
    "肾虚证": "肾脏精气不足。常见症状包括腰膝酸软、头晕耳鸣、记忆力减退、脱发、性功能减退等。偏阳虚则怕冷、小便清长；偏阴虚则手足心热、盗汗。",
    "肾气证": "肾气亏虚，固摄功能减弱。常见症状包括腰膝酸软、小便频数、夜尿增多、气短乏力、听力减退等。治疗以补肾益气为主。",
    "阴阳两虚": "阴虚和阳虚的症状同时出现。既有阴虚的潮热盗汗、口干咽燥，又有阳虚的畏寒怕冷、四肢不温。治疗需阴阳双补。",
    "痰湿证": "体内水液代谢失常，产生痰湿内停。常见症状包括身体困重、胸闷、痰多、口中黏腻、大便溏而不爽、舌苔厚腻等。治疗以化痰祛湿为主。",
    "血瘀证": "血液运行不畅或停滞。常见症状包括刺痛（固定不移）、面色晦暗、唇舌紫暗、皮肤瘀斑、月经有血块等。治疗以活血化瘀为主。",
    "阴虚火旺": "阴液亏虚导致虚火上炎。常见症状包括口干咽燥、潮热盗汗、五心烦热、失眠多梦、舌红少苔等。治疗以滋阴降火为主。",
    "阳虚水泛证": "阳气虚弱，不能温化水湿，水湿内停泛溢。常见症状包括水肿、小便不利、畏寒怕冷、腰膝酸软等。治疗以温阳利水为主。",
    "中焦虚寒证": "脾胃阳气不足，虚寒内生。常见症状包括胃脘隐痛（得温缓解）、食欲不振、喜热饮、大便稀溏等。治疗以温中散寒为主。",
    "血虚寒厥证": "血虚兼寒邪凝滞经络，四肢末端失于温养。常见症状包括手足冰冷（甚至厥冷）、麻木、面色苍白等。治疗以养血散寒、温通经络为主。",
    "肝郁腑实证": "肝气郁结兼胃肠实热积滞。常见症状包括胸胁胀痛、口苦口干、大便秘结、心烦易怒等。治疗需疏肝解郁、通腑泻热。",
    "痰火扰神证": "痰浊与火热互结，上扰心神。常见症状包括心烦失眠、多梦易惊、胸闷痰多、口苦、舌苔黄腻等。治疗以清热化痰、宁心安神为主。",
    # 方剂解释
    "川芎茶调散": "出自《太平惠民和剂局方》，主治外感风邪头痛。以川芎为君药，配羌活、白芷、细辛等祛风散寒、通络止痛。服时以清茶调下，取其清上降下之功。",
    "归脾汤": "出自《济生方》，主治心脾两虚、气血不足所致的心悸失眠、体倦食少。以人参、黄芪、白术健脾益气，当归、龙眼肉养血安神。",
    "六味地黄丸": "出自《小儿药证直诀》，滋补肾阴的经典方。以熟地黄为君，配山茱萸、山药补肝肾、益脾阴，泽泻、牡丹皮、茯苓泻浊清热。三补三泻，补而不滞。",
    "逍遥散": "出自《太平惠民和剂局方》，主治肝郁血虚脾弱证。以柴胡疏肝解郁，当归、白芍养血柔肝，白术、茯苓健脾和胃。疏肝解郁、养血健脾并重。",
    "真武汤": "出自《伤寒论》，主治阳虚水泛证。以附子温肾阳，茯苓、白术健脾利水，生姜温散水气，白芍敛阴和营。温阳与利水并行。",
    "小建中汤": "出自《伤寒论》，主治中焦虚寒、肝脾不和所致的腹痛。以饴糖为君甘温补中，配桂枝、白芍调和营卫，生姜、大枣、甘草补脾胃。",
    "桂枝汤": "出自《伤寒论》，「群方之祖」。主治外感风寒表虚证，以桂枝解肌发表、白芍敛阴和营，姜枣草调补脾胃。调和营卫，解肌发表。",
    "当归四逆汤": "出自《伤寒论》，主治血虚寒厥证。以当归、白芍养血和营，桂枝、细辛温经散寒，通草通利血脉，大枣、甘草补中益气。养血散寒、温通经脉。",
    "温经汤": "出自《金匮要略》，主治冲任虚寒、瘀血阻滞所致的月经不调。温经散寒与养血祛瘀并用，是妇科调经的经典方。",
    "酸枣仁汤": "出自《金匮要略》，主治肝血不足、虚热内扰所致的虚烦不眠。以酸枣仁养肝血、安心神，知母清虚热，川芎调肝血，茯苓宁心，甘草和中。",
}

# 中医养生建议模板
LIFESTYLE_ADVICE_TEMPLATES = {
    "风": "避风寒，注意保暖，尤其头部和颈部。适当运动如太极拳、八段锦以助气血流通。饮食宜清淡，多吃葱、姜、香菜等辛散之品。",
    "寒": "注意保暖，尤其腹部和足部。可常用热水泡脚（可加艾叶、生姜）。饮食宜温热，多吃羊肉、生姜、桂圆等温性食物。忌生冷寒凉之品。",
    "湿": "居住环境保持干燥通风。适当运动出汗以祛湿。饮食宜清淡，多吃薏米、赤小豆、冬瓜、山药等健脾祛湿之品。少吃甜腻、油腻食物。",
    "热": "保持心情平和，避免急躁。居室宜清凉通风。多吃绿豆、苦瓜、西瓜、莲子等清热之品。忌辛辣、煎炸、烧烤食物。",
    "虚": "注意休息，避免过度劳累。保证充足睡眠（晚上11点前入睡）。饮食宜营养丰富、易于消化。适当进补（根据气虚/血虚/阴虚/阳虚选用不同补品）。",
    "瘀": "保持心情舒畅，避免情绪抑郁。适当运动促进气血流通。可常饮玫瑰花茶、山楂茶。饮食宜清淡，少吃油腻。",
    "痰": "饮食清淡，少吃肥甘厚腻、甜食和奶制品。适当运动有助于化痰。可常饮陈皮茶、薏米水。保持大便通畅。",
    "气": "保持心情舒畅，学会自我减压。可练习深呼吸、冥想。适当运动如散步、瑜伽。饮食规律，避免暴饮暴食。",
}


def get_lifestyle_advice(diagnosis_text):
    """根据证型诊断生成生活调养建议"""
    advice_parts = []
    for keyword, advice in LIFESTYLE_ADVICE_TEMPLATES.items():
        if keyword in diagnosis_text:
            advice_parts.append(advice)
    if not advice_parts:
        advice_parts.append("保持规律作息，饮食均衡，心情舒畅，适当运动。具体建议请咨询专业中医师。")
    return ' '.join(advice_parts)


# ============================================================
# LLM 调用封装
# ============================================================
def call_llm(system_prompt, user_prompt, temperature=0.7, max_tokens=2000):
    """调用LLM（OpenAI兼容接口）"""
    if not USE_LLM:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM调用失败: {e}")
        return None


def llm_generate_followup(chief_complaint, candidate_count, sample_records, previous_qa, round_num):
    """使用LLM生成追问问题"""
    sample_texts = []
    for r in sample_records[:8]:
        symptoms = r.get('symptoms', '')[:300]
        diagnosis = r.get('diagnosis', '')[:100]
        sample_texts.append(f"病例症状：{symptoms}\n病例诊断：{diagnosis}")
    samples = '\n---\n'.join(sample_texts)
    prev_qa_text = '\n'.join([f"第{q['round']}轮追问：{q['question']} → 患者回答：{q['answer']}" for q in previous_qa]) if previous_qa else '（尚无追问记录）'
    system_prompt = """你是一位资深中医专家，正在进行多轮问诊。你的任务是基于患者的症状和候选病例，生成1个精准的追问问题来缩小诊断范围。
要求：
1. 问题应有中医辨证特色（关注寒热、虚实、表里、脏腑定位等）
2. 提供3-6个互斥的选项，方便患者选择
3. 指示该问题是单选还是多选
4. 输出严格按JSON格式：{"question": "...", "options": ["...", "..."], "multi_select": false}
只输出JSON，不要有其他文字。"""
    user_prompt = f"""患者主诉：{chief_complaint}
当前匹配到 {candidate_count} 条候选病例，需要进一步追问缩小范围。
当前是第{round_num}轮追问（最多5轮）。
历史追问与回答：
{prev_qa_text}
部分候选病例参考：
{samples}
请生成第{round_num}轮的追问问题（JSON格式）："""
    result = call_llm(system_prompt, user_prompt, temperature=0.7, max_tokens=400)
    if result:
        try:
            result = result.strip()
            if result.startswith('```'):
                result = result.split('\n', 1)[1]
                if result.endswith('```'):
                    result = result[:-3]
            return json.loads(result)
        except json.JSONDecodeError:
            logger.warning(f"LLM返回格式解析失败: {result[:200]}")
    return None


def llm_generate_diagnosis(chief_complaint, age, gender, basic_info, followup_history, matched_records):
    """使用LLM生成最终诊断"""
    # 准备参考病例摘要
    record_summaries = []
    for r in matched_records[:10]:
        diag = (r.get('diagnosis', '') or '').strip()
        formula = (r.get('formula', '') or '').strip()
        herbs = (r.get('herbs', '') or '').strip()
        symptoms = (r.get('symptoms', '') or '')[:400]
        record_summaries.append(f"【病例】症状：{symptoms}\n诊断：{diag}\n方剂：{formula}\n中药：{herbs}")
    records_text = '\n---\n'.join(record_summaries)
    qa_text = '\n'.join([f"医生追问：{q['question']}\n患者回答：{q['answer']}" for q in followup_history]) if followup_history else ''

    system_prompt = """你是一位经验丰富的中医专家——"孙思邈"。请基于患者信息和参考病例，给出专业的中医诊断建议。
你必须使用规范的中医专业用语（如"腰膝酸软"、"肝气郁结"、"舌淡苔白"、"脉弦细"等）。

请按以下格式输出：

1. 辨证分型：
（给出中医证型诊断，如"风寒袭络证"、"肝郁脾虚证"等，并简要说明辨证依据）

2. 参考方剂：
（给出推荐方剂名称，如"川芎茶调散加减"）

3. 具体中药及剂量：
（逐行列出中药名称和剂量，格式：中药名+剂量（克），如"川芎10g、白芷10g"）

4. 生活调养建议：
（给出具体可行的饮食、起居、情志、运动等方面建议）

5. 中医理论依据：
（用通俗易懂的语言解释为什么这样治疗，引用中医基础理论）

注意：所有建议仅供参考，实际用药请在专业中医师指导下进行。"""

    user_prompt = f"""请为以下患者进行中医诊断：

【患者基本信息】
年龄：{age}岁
性别：{gender}
身高：{basic_info.get('height', '未知')}
体重：{basic_info.get('weight', '未知')}

【主诉】
{chief_complaint}

【问诊过程】
{qa_text if qa_text else '（无追问记录）'}

【匹配的参考病例】（共{len(matched_records)}条）
{records_text}

请给出完整的中医诊断建议："""

    result = call_llm(system_prompt, user_prompt, temperature=0.7, max_tokens=2000)
    return result


def llm_explain_term(term):
    """使用LLM解释中医术语"""
    system_prompt = """你是一位耐心的中医科普专家。请用通俗易懂、深入浅出的语言解释中医术语。
要求：
1. 用老百姓能听懂的大白话解释
2. 可以用比喻、类比帮助理解
3. 简要说明该术语在临床上的意义
4. 200-400字为宜"""
    user_prompt = f"请解释以下中医术语的含义：{term}"
    result = call_llm(system_prompt, user_prompt, temperature=0.5, max_tokens=600)
    return result


# ============================================================
# 规则引擎：追问问题生成（LLM不可用时的后备方案）
# ============================================================
def rule_based_followup(chief_complaint, records, previous_qa, round_num):
    """规则引擎生成追问问题"""
    # 获取匹配的问题模板
    question_templates = get_followup_questions_for_complaint(chief_complaint)
    # 过滤掉已经在之前问过同类问题
    asked_categories = set()
    for qa in previous_qa:
        asked_categories.add(qa.get('category', ''))
    available = [q for q in question_templates if q.get('category', '') not in asked_categories]
    if not available:
        available = [q for q in GENERIC_FOLLOWUP_QUESTIONS if q.get('category', '') not in asked_categories]
    if not available:
        # 所有问题都问过了
        return None
    # 选择当前轮次的问题
    idx = min(round_num - 1, len(available) - 1)
    selected = available[idx]
    options = generate_dynamic_options(selected, records, chief_complaint)
    if not options:
        options = selected.get('options', ['是', '否'])
    return {
        "question": selected['question'],
        "options": options,
        "multi_select": selected.get('multi_select', False),
        "category": selected.get('category', ''),
    }


def rule_based_diagnosis(chief_complaint, age, gender, basic_info, matched_records):
    """规则引擎生成诊断建议（基于病例统计分析）"""
    features = extract_symptom_features(matched_records)
    top_diag = features['common_diagnoses'][0] if features['common_diagnoses'] else ('待进一步辨证', 0)
    top_formula = features['common_formulas'][0] if features['common_formulas'] else ('待进一步选方', 0)
    top_herbs = features['common_herbs'][:10]

    diagnosis_name = top_diag[0] if isinstance(top_diag, tuple) else top_diag
    formula_name = top_formula[0] if isinstance(top_formula, tuple) else top_formula

    # 从病例中收集实际方剂和中药
    herb_dosage_map = defaultdict(list)
    for r in matched_records[:10]:
        herbs_text = (r.get('herbs', '') or '').strip()
        if herbs_text:
            for part in herbs_text.split():
                match = re.match(r'([一-鿿]+)([\d.]+)', part)
                if match:
                    herb_name = match.group(1)
                    dosage = float(match.group(2))
                    herb_dosage_map[herb_name].append(dosage)

    herb_lines = []
    for herb_name, dosages in herb_dosage_map.items():
        avg_dose = sum(dosages) / len(dosages)
        herb_lines.append(f"{herb_name}{avg_dose:.0f}g" if avg_dose == int(avg_dose) else f"{herb_name}{avg_dose:.1f}g")

    if not herb_lines:
        herb_lines = [f"{h[0]}" for h in top_herbs[:8]]

    # 统计常见的伴随症状
    common_symptoms = [s[0] for s in features['common_symptoms'][:10] if len(s[0]) >= 2]

    # 生活调养建议
    lifestyle = get_lifestyle_advice(diagnosis_name)

    # 中医理论依据（基于证型特征）
    theory = generate_theory_basis(diagnosis_name, common_symptoms)

    diagnosis_text = f"""1. 辨证分型：
{diagnosis_name}
（基于{len(matched_records)}例相似病例的统计分析，此为最常见证型。相似病例中常见症状包括：{'、'.join(common_symptoms[:8])}）

2. 参考方剂：
{formula_name}加减
（在匹配的{len(matched_records)}例病例中，该方剂使用频率最高）

3. 具体中药及剂量：
{'、'.join(herb_lines[:12])}
（以上为相似病例中高频使用的中药及常用剂量，具体用量需根据个体情况调整）

4. 生活调养建议：
{lifestyle}

5. 中医理论依据：
{theory}

⚠️ 注意：以上为基于数据统计的参考建议，非AI实时诊断。如需更精准的辨证施治，请配置LLM API后重试。实际用药请在专业中医师指导下进行。"""

    return diagnosis_text


def generate_theory_basis(diagnosis_name, common_symptoms):
    """根据证型和症状生成中医理论依据"""
    theory_parts = []
    if '风' in diagnosis_name:
        theory_parts.append('风为百病之长，善行而数变，风邪外袭首犯肌表经络。中医认为「治风先治血，血行风自灭」，故治疗需祛风与养血并行。')
    if '寒' in diagnosis_name:
        theory_parts.append('寒为阴邪，易伤阳气，其性收引凝滞，导致气血运行不畅。「寒者热之」是基本治疗原则，以温阳散寒为主。')
    if '湿' in diagnosis_name:
        theory_parts.append('湿性重浊黏腻，易阻滞气机，困遏脾胃。脾主运化水湿，故「治湿不理脾，非其治也」，健脾祛湿为治疗关键。')
    if '热' in diagnosis_name or '火' in diagnosis_name:
        theory_parts.append('热（火）为阳邪，其性炎上，易伤津耗气，扰动心神。「热者寒之」，治疗以清热泻火、生津养阴为主。')
    if '虚' in diagnosis_name:
        theory_parts.append('「虚则补之」是中医治疗虚证的基本原则。人体之虚有气血阴阳之不同，需辨证施补，虚什么补什么，不可盲目进补。')
    if '瘀' in diagnosis_name:
        theory_parts.append('瘀血既是病理产物，又是致病因素。「不通则痛」，气血瘀滞可导致疼痛、肿块等症状。治疗以活血化瘀、通络止痛为主。')
    if '痰' in diagnosis_name:
        theory_parts.append('「百病多由痰作祟」，痰之为病随气升降无处不到。脾为生痰之源，肺为贮痰之器，故治痰需健脾化痰、理气行滞。')
    if '肝' in diagnosis_name:
        theory_parts.append('肝主疏泄，喜条达而恶抑郁。肝脏功能失调可影响全身气机运行，进而导致多种病症。')
    if '脾' in diagnosis_name:
        theory_parts.append('脾胃为后天之本，气血生化之源。脾主运化升清，胃主受纳降浊。脾胃功能正常是维持全身健康的基础。')
    if '肾' in diagnosis_name:
        theory_parts.append('肾为先天之本，主藏精、主水、主纳气。肾中精气是人体生命活动的根本动力，五脏之病久必及肾。')
    if not theory_parts:
        theory_parts.append('中医讲究「辨证论治」，即根据患者的具体症状、体征，综合分析病位、病性、病势，确定证型，然后依证立法、依法选方、据方用药。这体现了中医「同病异治、异病同治」的个体化诊疗特点。')
    return ' '.join(theory_parts)


def rule_based_explain(term):
    """规则引擎解释中医术语"""
    # 先查词典
    if term in TCM_TERM_DICT:
        return TCM_TERM_DICT[term]
    # 模糊匹配
    for key, value in TCM_TERM_DICT.items():
        if term in key or key in term:
            return f'关于「{term}」的解释：\n{value}'
    # 通用解释模板
    return f'''「{term}」是一个中医专业术语。

在中医理论中，这类术语通常涉及对人体生理、病理状态的独特描述。中医使用「辨证论治」的方法——通过望、闻、问、切四诊收集信息，运用八纲辨证（阴阳、表里、寒热、虚实）、脏腑辨证等方法，综合分析得出结论。

如果您能提供该术语出现的上下文（如在什么病症描述中出现的），我可以给出更精准的解释。您也可以咨询专业中医师获取权威解读。'''


# ============================================================
# 静态文件服务
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_FILE = 'index.html'
LLM_HTML_FILE = 'llm.html'
HUNHE_HTML_FILE = '混合版.html'

@app.route('/')
def index():
    """专业版 — 无需 API Key"""
    return send_from_directory(BASE_DIR, HTML_FILE)

@app.route('/hunhe')
def hunhe_index():
    """混合版 — 大众版 AI 诊疗"""
    return send_from_directory(BASE_DIR, HUNHE_HTML_FILE)

@app.route('/hunhe-pro')
def hunhe_pro_index():
    """混合版 · 专业版检索界面（含大众版返回键）"""
    return send_from_directory(BASE_DIR, '混合版专业.html')

@app.route('/llm')
@app.route('/dazhong')
def llm_index():
    """大众版 — AI 智能诊疗"""
    return send_from_directory(BASE_DIR, LLM_HTML_FILE)

@app.route('/<path:filename>')
def serve_static(filename):
    """提供所有静态文件（data/*.json, images/*.jpg 等）"""
    # 安全检查：防止目录遍历
    safe_path = os.path.normpath(os.path.join(BASE_DIR, filename))
    if not safe_path.startswith(os.path.normpath(BASE_DIR)):
        return jsonify({'error': '禁止访问'}), 403
    if os.path.isfile(safe_path):
        return send_from_directory(BASE_DIR, filename)
    return jsonify({'error': '文件不存在'}), 404

# ============================================================
# API 路由
# ============================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'total_records': len(all_records) if _data_loaded else 'lazy_loading',
        'llm_enabled': USE_LLM,
    })


@app.route('/api/start_consultation', methods=['POST'])
def start_consultation():
    """
    步骤1+2：接收基础信息和主诉，开始问诊
    返回候选病例数量和第一轮追问问题
    """
    load_all_data()  # 按需加载数据
    data = request.get_json()
    if not data:
        return jsonify({'error': '请求数据为空'}), 400

    name = data.get('name', '')
    gender = data.get('gender', '')
    age = data.get('age')
    chief_complaint = data.get('chief_complaint', '').strip()
    height = data.get('height', '')
    weight = data.get('weight', '')

    # 验证必填项
    if not gender or gender not in ['男', '女']:
        return jsonify({'error': '请选择性别'}), 400
    if not age or not str(age).isdigit():
        return jsonify({'error': '请填写有效年龄'}), 400
    if not chief_complaint:
        return jsonify({'error': '请输入主要症状（主诉）'}), 400

    age = int(age)
    # 搜索匹配病例
    matched = search_by_symptom(chief_complaint, gender=gender, age=age)
    candidate_count = len(matched)

    # 生成会话ID
    session_id = str(uuid.uuid4())

    # 存储会话状态
    session_store[session_id] = {
        'name': name,
        'gender': gender,
        'age': age,
        'height': height,
        'weight': weight,
        'chief_complaint': chief_complaint,
        'matched_records': matched,
        'candidate_count': candidate_count,
        'followup_history': [],
        'round': 0,
        'max_rounds': 5,
        'threshold': 20,
    }

    response = {
        'session_id': session_id,
        'candidate_count': candidate_count,
        'need_followup': candidate_count > 20,
        'message': f'检索到 {candidate_count} 条相关病例',
    }

    # 如果候选病例 > 20，需要追问
    if candidate_count > 20:
        question = None
        if USE_LLM:
            question = llm_generate_followup(
                chief_complaint, candidate_count,
                matched[:15], [], 1
            )
        if not question:
            question = rule_based_followup(chief_complaint, matched, [], 1)
        if question:
            response['followup_question'] = question
            response['round'] = 1
            session_store[session_id]['round'] = 1
        else:
            response['need_followup'] = False
    else:
        # 直接生成诊断
        basic_info = {'height': height, 'weight': weight}
        diagnosis = None
        if USE_LLM:
            diagnosis = llm_generate_diagnosis(
                chief_complaint, age, gender, basic_info, [], matched
            )
        if not diagnosis:
            diagnosis = rule_based_diagnosis(chief_complaint, age, gender, basic_info, matched)
        response['diagnosis'] = diagnosis

    return jsonify(response)


@app.route('/api/answer_followup', methods=['POST'])
def answer_followup():
    """
    步骤3：接收追问回答，返回下一轮追问或最终诊断
    """
    load_all_data()  # 按需加载数据
    data = request.get_json()
    if not data:
        return jsonify({'error': '请求数据为空'}), 400

    session_id = data.get('session_id', '')
    answer = data.get('answer', [])
    if isinstance(answer, str):
        answer = [answer]

    if session_id not in session_store:
        return jsonify({'error': '会话已过期，请重新开始问诊'}), 400

    state = session_store[session_id]
    current_round = state['round']
    chief_complaint = state['chief_complaint']
    gender = state['gender']
    age = state['age']
    name = state['name']
    height = state['height']
    weight = state['weight']

    # 记录本轮的追问和回答
    last_question = data.get('question_text', '')
    state['followup_history'].append({
        'round': current_round,
        'question': last_question,
        'answer': '、'.join(answer) if answer else '',
        'category': data.get('category', ''),
    })

    # 根据回答筛选病例
    if answer:
        additional_filters = {}
        for ans in answer:
            ans_lower = ans.lower().strip()
            additional_filters[ans_lower] = ans_lower
        state['matched_records'] = [
            r for r in state['matched_records']
            if any(ans.lower() in (r.get('symptoms', '') or '').lower() for ans in answer)
        ]
        # 如果过滤后太少，回退到不过滤
        if len(state['matched_records']) < 3:
            # 使用宽松匹配
            state['matched_records'] = search_by_symptom(
                chief_complaint, gender=gender, age=age,
                additional_filters={'answers': answer} if answer else None
            )
            if len(state['matched_records']) < 3:
                # 完全回退
                state['matched_records'] = search_by_symptom(chief_complaint, gender=gender, age=age)

    candidate_count = len(state['matched_records'])
    state['candidate_count'] = candidate_count

    response = {
        'session_id': session_id,
        'candidate_count': candidate_count,
        'round': current_round,
        'message': f'筛选后剩余 {candidate_count} 条候选病例',
    }

    # 判断是否需要继续追问
    if candidate_count > state['threshold'] and current_round < state['max_rounds']:
        next_round = current_round + 1
        state['round'] = next_round

        question = None
        if USE_LLM:
            question = llm_generate_followup(
                chief_complaint, candidate_count,
                state['matched_records'][:15],
                state['followup_history'],
                next_round
            )
        if not question:
            question = rule_based_followup(
                chief_complaint,
                state['matched_records'],
                state['followup_history'],
                next_round
            )

        if question:
            response['followup_question'] = question
            response['round'] = next_round
            response['need_followup'] = True
        else:
            response['need_followup'] = False
            # 所有问题问完，直接诊断
            basic_info = {'height': height, 'weight': weight}
            diagnosis = None
            if USE_LLM:
                diagnosis = llm_generate_diagnosis(
                    chief_complaint, age, gender, basic_info,
                    state['followup_history'], state['matched_records']
                )
            if not diagnosis:
                diagnosis = rule_based_diagnosis(
                    chief_complaint, age, gender, basic_info,
                    state['matched_records']
                )
            response['diagnosis'] = diagnosis
    else:
        response['need_followup'] = False
        # 达到阈值或达到最大轮数，生成诊断
        basic_info = {'height': height, 'weight': weight}
        diagnosis = None
        if USE_LLM:
            diagnosis = llm_generate_diagnosis(
                chief_complaint, age, gender, basic_info,
                state['followup_history'], state['matched_records']
            )
        if not diagnosis:
            diagnosis = rule_based_diagnosis(
                chief_complaint, age, gender, basic_info,
                state['matched_records']
            )
        response['diagnosis'] = diagnosis

    return jsonify(response)


@app.route('/api/explain_term', methods=['POST'])
def explain_term():
    """
    步骤5：解释中医术语
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': '请求数据为空'}), 400

    term = data.get('term', '').strip()
    if not term:
        return jsonify({'error': '请输入要解释的术语'}), 400

    explanation = None
    if USE_LLM:
        explanation = llm_explain_term(term)
    if not explanation:
        explanation = rule_based_explain(term)

    return jsonify({
        'term': term,
        'explanation': explanation,
    })


@app.route('/api/records_count', methods=['GET'])
def records_count():
    """获取数据统计"""
    load_all_data()
    return jsonify({
        'total_records': len(all_records),
        'data_files': NUM_DATA_FILES,
        'sample_record': all_records[0] if all_records else None,
    })


def _search_file_text(filepath, query, max_results):
    """文本方式搜索：找到匹配片段，只 json.loads 那几条记录（内存极省）"""
    results = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        pos = 0
        while pos < len(text) and len(results) < max_results:
            idx = text.find(query, pos)
            if idx == -1:
                break
            # 向前找最近的 {
            brace = text.rfind('{', 0, idx)
            if brace == -1:
                pos = idx + len(query)
                continue
            # 向后匹配完整的 {} 块
            depth = 0
            end = brace
            for j in range(brace, min(brace + 5000, len(text))):
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break
            if end > brace:
                try:
                    rec = json.loads(text[brace:end])
                    sym = (rec.get('symptoms', '') or '').lower()
                    diag = (rec.get('diagnosis', '') or '').lower()
                    fm = (rec.get('formula', '') or '').lower()
                    if query in sym or query in diag or query in fm:
                        results.append({
                            'symptoms': (rec.get('symptoms', '') or '')[:500],
                            'diagnosis': (rec.get('diagnosis', '') or '')[:200],
                            'formula': (rec.get('formula', '') or '')[:200],
                            'herbs': (rec.get('herbs', '') or '')[:500],
                        })
                except json.JSONDecodeError:
                    pass
            pos = end if end > brace else idx + len(query)
        del text
    except Exception as e:
        logger.error(f'搜索 {os.path.basename(filepath)} 出错: {e}')
    return results


@app.route('/api/search', methods=['POST'])
def search_records():
    """
    服务端搜索医案数据（逐文件加载+立即释放，安全高效）
    """
    data = request.get_json()
    if not data or not data.get('query', '').strip():
        return jsonify({'error': '查询内容为空'}), 400

    query = data['query'].strip().lower()
    max_results = data.get('max_results', 20)
    results = []

    import gc
    for i in range(1, NUM_DATA_FILES + 1):
        filepath = os.path.join(DATA_DIR, f'medical_records_part{i}.json')
        if not os.path.exists(filepath):
            continue
        new_results = _search_file_text(filepath, query, max_results - len(results))
        results.extend(new_results)
        gc.collect()
        if len(results) >= max_results:
            break

    return jsonify({
        'results': results,
        'count': len(results),
        'query': query,
    })


@app.route('/api/formula', methods=['POST'])
def lookup_formula():
    """
    根据方剂名称查找详细信息（文本搜索，只解析匹配记录）
    """
    data = request.get_json()
    if not data or not data.get('name', '').strip():
        return jsonify({'error': '方剂名称为空'}), 400

    fname = data['name'].strip()
    for i in range(1, NUM_DATA_FILES + 1):
        filepath = os.path.join(DATA_DIR, f'medical_records_part{i}.json')
        if not os.path.exists(filepath):
            continue
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            idx = text.find(fname)
            if idx == -1:
                del text
                continue
            brace = text.rfind('{', 0, idx)
            if brace == -1:
                del text
                continue
            depth = 0
            end = brace
            for j in range(brace, min(brace + 5000, len(text))):
                if text[j] == '{': depth += 1
                elif text[j] == '}':
                    depth -= 1
                    if depth == 0: end = j + 1; break
            if end > brace:
                rec = json.loads(text[brace:end])
                if fname in (rec.get('formula', '') or ''):
                    del text
                    return jsonify({
                        'found': True, 'name': fname,
                        'herbs': (rec.get('herbs', '') or '')[:500],
                        'diagnosis': (rec.get('diagnosis', '') or '')[:200],
                        'symptoms': (rec.get('symptoms', '') or '')[:500],
                    })
            del text
        except Exception as e:
            logger.error(f'查找方剂 {os.path.basename(filepath)} 出错: {e}')

    return jsonify({'found': False, 'name': fname})


@app.route('/api/chat', methods=['POST'])
def ai_chat():
    """
    AI 聊天接口
    - 规则版（默认）：优先 LLM，无 Key 时自动切换规则引擎
    - LLM 版（force_llm=true）：必须有 Key，否则返回错误
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': '请求数据为空'}), 400

    messages = data.get('messages', [])
    if not messages:
        return jsonify({'error': '对话内容为空'}), 400

    force_llm = data.get('force_llm', False)

    # 提取用户最后一条消息
    user_msg = ''
    for m in reversed(messages):
        if m.get('role') == 'user':
            user_msg = m.get('content', '')
            break

    # 获取 API Key（服务端环境变量 > 用户个人 Key）
    api_key = LLM_API_KEY or data.get('api_key', '')

    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=LLM_BASE_URL)
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )
            reply = response.choices[0].message.content
            return jsonify({'reply': reply, 'model': LLM_MODEL, 'mode': 'llm'})
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            if force_llm:
                return jsonify({'error': f'大模型调用失败: {str(e)}。请检查 API Key 是否正确。'}), 500
            # 规则版继续走规则引擎

    # force_llm 模式但无可用 Key
    if force_llm:
        return jsonify({
            'error': '未配置有效的 API Key。请在页面输入 DeepSeek API Key，或联系管理员在服务端配置 LLM_API_KEY 环境变量。'
        }), 400

    # 规则引擎后备 — 无 Key 也能用
    reply = rule_based_chat(user_msg, messages)
    return jsonify({'reply': reply, 'model': 'rule-engine', 'mode': 'rule'})


# ============================================================
# 规则引擎聊天回复
# ============================================================

# 危险信号关键词 → 紧急就医建议
DANGER_SIGNALS = {
    '雷击': '⚡ 您描述的"雷击样剧烈头痛"是需要高度警惕的危险信号！这可能是蛛网膜下腔出血的征兆。\n\n🚨 请立即前往医院急诊科就诊，不要延误！',
    '爆炸': '⚡ 您描述的"爆炸性头痛"属于危险信号！需排除蛛网膜下腔出血等急症。\n\n🚨 建议立即就医检查。',
    '口眼歪斜': '🔴 面部歪斜+肢体无力是脑卒中的典型表现！\n\n🚨 请立即拨打120或前往最近医院急诊！',
    '肢体无力': '🔴 伴有肢体无力的症状需警惕脑血管意外！\n\n🚨 请尽快就医，进行神经系统检查。',
    '言语不清': '🔴 言语不清可能是脑卒中的前兆！\n\n🚨 请立即就医，时间就是大脑。',
    '脖子硬': '🔴 头痛+颈部僵硬+发烧，需警惕脑膜炎！\n\n🚨 请立即前往医院检查。',
    '颈部僵硬': '🔴 头痛+颈部僵硬+发烧，需警惕脑膜炎！\n\n🚨 请立即前往医院检查。',
    '外伤': '🔴 外伤后的头痛需要排除颅内出血的可能！\n\n🚨 建议尽快去医院做头部CT检查。',
    '胸痛': '🔴 胸痛放射至左臂或下颌+冷汗，高度疑似心梗！\n\n🚨 请立即拨打120！',
    '冷汗': '🔴 胸痛伴冷汗是危险信号！\n\n🚨 请立即停止活动，拨打120。',
}

SYMPTOM_PATTERN_MAP = {
    '头痛|头疼|偏头痛': {
        '搏动|跳痛|一跳': '🟢 搏动性头痛常见于偏头痛。属于中医"头风"范畴，多因肝阳上亢或风邪上扰所致。',
        '紧箍|发紧|绷紧|束缚': '🟢 紧箍样头痛多见于紧张性头痛。中医认为多与肝郁气滞、颈部筋脉拘急有关。',
        '低头|久坐|伏案|脖子': '🟢 低头加重提示颈源性头痛！颈部肌肉长期紧张牵拉头部筋膜所致。',
        '鼻塞|流涕|鼻': '🟢 伴有鼻部症状的头痛多属鼻源性。中医认为与肺气不宣、鼻窍不通有关。',
        '怕冷|怕风|恶寒': '🟢 遇冷加重属风寒头痛，风寒之邪侵袭经络所致。',
        'default': '🟢 头痛的病因多样。中医将其分为外感头痛（风寒/风热/风湿）和内伤头痛（肝阳/血虚/痰浊/瘀血）。'
    },
    '失眠|睡不着|入睡|多梦|易醒|早醒': {
        '心烦|烦热|烦躁': '🟢 心烦失眠多属阴虚火旺或痰火扰神。虚火扰心，心神不宁所致。',
        '多梦|噩梦': '🟢 多梦易醒多与心血不足或肝血亏虚有关。血不养心，神魂不安。',
        '疲倦|乏力|没精神': '🟢 疲劳伴失眠多属心脾两虚。思虑过度，暗耗心血。',
        'default': '🟢 失眠在中医称为"不寐"。核心病机是阳不入阴，心神失养或心神被扰。常见证型有心脾两虚、阴虚火旺、痰热扰心、肝郁化火等。'
    },
    '胃痛|胃胀|腹痛|腹胀|消化': {
        '隐痛|绵绵|喜按': '🟢 隐痛喜按属虚证，多因脾胃虚寒或胃阴不足。得温得按则舒。',
        '胀痛|胀满|打嗝|嗳气': '🟢 胀痛属气滞，多因肝气犯胃或饮食积滞。气机不畅所致。',
        '灼热|烧心|反酸': '🟢 灼热感属胃热或肝胃郁热。热邪灼伤胃络所致。',
        'default': '🟢 胃痛中医称"胃脘痛"。常见证型：脾胃虚寒、肝气犯胃、饮食积滞、胃阴不足。'
    },
    '关节|膝盖|腰酸|腰膝|四肢': {
        '怕冷|得温|遇冷': '🟢 遇冷加重的关节疼痛多为风寒湿痹。寒性收引凝滞，气血不通则痛。',
        '红肿|发热|灼热': '🟢 红肿热痛属热痹，多因湿热之邪痹阻关节。',
        'default': '🟢 关节疼痛在中医属"痹证"范畴。风寒湿三气杂至合而为痹。治疗以祛风散寒除湿、活血通络止痛为主。'
    },
    '咳嗽|咳痰|感冒|咽': {
        '白痰|清稀|泡沫': '🟢 白痰清稀多为风寒或痰湿。',
        '黄痰|黏稠': '🟢 黄痰黏稠属风热犯肺或痰热壅肺。',
        '干咳|无痰|痒': '🟢 干咳无痰或咽痒属风燥伤肺或肺阴亏虚。',
        'default': '🟢 咳嗽为肺气上逆所致。外感咳嗽分风寒、风热、风燥；内伤咳嗽分痰湿、痰热、肝火、阴虚。'
    },
}

ACUPOINT_RECOMMENDATIONS = {
    '头痛|头疼': '💆 推荐穴位：\n• 太阳穴：眉梢和外眼角之间向后一横指凹陷处，用拇指按揉2-3分钟\n• 风池穴：后颈部两条大筋外缘凹陷处，双手拇指同时按压\n• 合谷穴：手背第1、2掌骨间，按压至有酸胀感',
    '失眠|睡不着': '💆 推荐穴位：\n• 神门穴：手腕横纹尺侧端凹陷处，睡前按揉3分钟\n• 三阴交：内踝尖上3寸（四横指），睡前按揉\n• 涌泉穴：足底前1/3凹陷处，热水泡脚后按揉',
    '胃|消化|腹胀|腹痛': '💆 推荐穴位：\n• 足三里：膝盖外侧凹陷下3寸（四横指），按揉5分钟\n• 中脘穴：肚脐上4寸（五横指），顺时针轻揉\n• 内关穴：手腕横纹上2寸（三横指），按压至酸胀',
    '关节|膝盖|腰|四肢': '💆 推荐穴位：\n• 阳陵泉：膝盖外侧下方凹陷处，按揉至温热感\n• 肾俞穴：腰部第2腰椎旁开1.5寸，搓热双手后按揉\n• 委中穴：膝盖后方腘窝横纹中点',
    '咳嗽|感冒|咽': '💆 推荐穴位：\n• 列缺穴：手腕横纹上1.5寸，按压可宣肺止咳\n• 天突穴：胸骨上窝正中，轻按止咽痒\n• 肺俞穴：背部第3胸椎旁开1.5寸',
}

def rule_based_chat(user_msg, messages):
    """无 LLM 时的规则引擎聊天回复 — 基于中医知识库"""
    msg = user_msg.strip()

    # 1. 危险信号排查（最高优先级）
    for keyword, warning in DANGER_SIGNALS.items():
        if keyword in msg:
            return warning + '\n\n（以上为系统自动识别的危险信号提示，请务必重视。以下为辅助参考信息）\n\n⚠️ 在任何情况下，如症状持续加重或出现新的严重症状，都应及时就医。'

    # 2. 构建回复
    reply_parts = []

    # 匹配症状模式
    matched_advice = []
    for pattern, sub_patterns in SYMPTOM_PATTERN_MAP.items():
        if re.search(pattern, msg):
            for sub_pat, advice in sub_patterns.items():
                if sub_pat == 'default':
                    continue
                if re.search(sub_pat, msg):
                    matched_advice.append(advice)
                    break
            # 如果没有匹配到子模式，用默认建议
            if not any(a for a in matched_advice if a and not a.startswith('🟢 头痛的病因')):
                default = sub_patterns.get('default', '')
                if default:
                    matched_advice.append(default)
            break  # 只匹配第一个命中的症状大类

    if matched_advice:
        reply_parts.append('【症状分析】')
        reply_parts.extend(matched_advice)

    # 3. 推荐穴位
    for pattern, acupoints in ACUPOINT_RECOMMENDATIONS.items():
        if re.search(pattern, msg):
            reply_parts.append('\n' + acupoints)
            break
    else:
        # 通用穴位
        reply_parts.append('\n💆 日常保健穴位：\n• 足三里（膝盖下3寸）— 健脾益气，强壮体质\n• 合谷穴（手背虎口处）— 通则不痛\n• 内关穴（手腕上2寸）— 宁心安神')

    # 4. 食疗建议
    reply_parts.append('\n🍵 饮食调理建议：')
    if re.search('寒|冷|怕冷|怕风', msg):
        reply_parts.append('• 宜食温热：生姜红糖水、桂圆红枣茶\n• 避食生冷：冰饮、西瓜、苦瓜等寒性食物')
    elif re.search('热|火|烧心|灼热|红肿|口干|口苦', msg):
        reply_parts.append('• 宜食清热：绿豆、冬瓜、菊花茶、梨\n• 避食辛辣：辣椒、羊肉、酒类等助热之品')
    elif re.search('湿|重|胀|腻|便溏|黏', msg):
        reply_parts.append('• 宜食祛湿：薏米红豆汤、山药、茯苓\n• 避食肥甘：甜腻点心、油炸食品')
    else:
        reply_parts.append('• 清淡饮食，避免辛辣油腻\n• 保持三餐规律，勿暴饮暴食\n• 适量饮水，促进新陈代谢')

    # 5. 生活建议
    reply_parts.append('\n💡 生活调养：')
    if re.search('失眠|睡', msg):
        reply_parts.append('• 晚上11点前入睡，睡前1小时不看手机\n• 温水泡脚15-20分钟，可加艾叶\n• 白天适当运动，但睡前2小时避免剧烈运动')
    elif re.search('头痛|头', msg):
        reply_parts.append('• 避免长时间低头看手机\n• 工作间隙做颈部拉伸：收下巴、侧向拉伸\n• 保持充足睡眠，避免熬夜')
    elif re.search('胃|消化|腹胀', msg):
        reply_parts.append('• 饭后散步15分钟，促进消化\n• 细嚼慢咽，每餐七分饱\n• 保持心情舒畅，避免生气时进食')
    else:
        reply_parts.append('• 规律作息，保证充足睡眠\n• 适当运动（散步、八段锦、太极拳）\n• 保持心情舒畅，避免过度焦虑')

    # 6. 追问
    reply_parts.append('\n❓ 为了更精准地分析，请补充：')
    if '头痛' in msg or '头疼' in msg:
        reply_parts.append('• 疼痛的具体部位在哪里？（前额/太阳穴/头顶/后脑勺/整个头部）\n• 疼痛是什么感觉？（搏动性跳痛/紧箍样压迫感/胀痛/刺痛）')
    elif '失眠' in msg or '睡' in msg:
        reply_parts.append('• 主要表现为入睡困难、睡后易醒、还是早醒？\n• 做梦多吗？白天精神如何？')
    elif '胃' in msg or '腹' in msg or '消化' in msg:
        reply_parts.append('• 疼痛和饮食有关系吗？（饭前痛/饭后痛/饥饿时痛）\n• 大便情况如何？（正常/便秘/便溏/黏腻）')
    elif '关节' in msg or '腰' in msg or '膝' in msg:
        reply_parts.append('• 是固定位置痛还是游走性疼痛？\n• 遇冷加重还是遇热加重？')
    else:
        reply_parts.append('• 这个症状持续多长时间了？\n• 有没有其他伴随的不适？（如怕冷、出汗、口干、乏力等）')

    reply_parts.append('\n🌿 以上分析基于中医理论，仅供参考。如症状持续或加重，请及时就医。')

    return '\n'.join(reply_parts)


# ============================================================
# 启动入口
# ============================================================
if __name__ == '__main__':
    import sys
    # Render 环境检测：Render 始终设置 RENDER=true
    is_render = os.environ.get('RENDER', 'false').lower() == 'true'
    is_prod = is_render or os.environ.get('FLASK_ENV') == 'production'

    print("=" * 60)
    print("  中医药辅助诊疗系统")
    print("=" * 60)
    print(f"  医案数据: 按需加载（启动时跳过以节省内存）")
    print(f"  LLM: {'DeepSeek (' + LLM_MODEL + ')' if USE_LLM else '规则引擎（无需 API Key）'}")
    print()
    print(f"  📋 专业版（无需 Key）: http://127.0.0.1:5000/")
    print(f"  🧬 LLM 大模型版（需 Key）:  http://127.0.0.1:5000/llm")
    print("=" * 60)

    host = '0.0.0.0' if is_prod else '127.0.0.1'
    port = int(os.environ.get('PORT', 5000))
    print(f"启动: host={host}, port={port}, is_prod={is_prod}, is_render={is_render}")
    app.run(host=host, port=port, debug=not is_prod)
