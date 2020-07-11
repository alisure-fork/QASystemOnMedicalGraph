import os
import re
import jieba
import string
import numpy as np
import ahocorasick  # AC自动机
from py2neo import Graph
from sklearn.externals import joblib
from gensim.models import KeyedVectors


class EntityExtractor:

    def __init__(self, data_dir="./data"):
        self.vocab_path = os.path.join(data_dir, 'vocab.txt')
        jieba.load_userdict(self.vocab_path)
        self.word2vec_path = os.path.join(data_dir, 'merge_sgns_bigram_char300.txt')
        self.model = KeyedVectors.load_word2vec_format(self.word2vec_path, binary=False)

        self.stopwords_path =os.path.join(data_dir, 'stop_words.utf8')
        self.stopwords = [w.strip() for w in open(self.stopwords_path, 'r', encoding='utf8') if w.strip()]

        # 意图分类模型文件  朴素贝叶斯模型
        self.tfidf_path = 'model/tfidf_model.m'
        self.tfidf_model = joblib.load(self.tfidf_path)
        self.nb_path =  'model/intent_reg_model.m'
        self.nb_model = joblib.load(self.nb_path)

        # 实体
        self.disease_path = os.path.join(data_dir, 'disease_vocab.txt')
        self.disease_entities = [w.strip() for w in open(self.disease_path, encoding='utf8') if w.strip()]
        self.alias_path = os.path.join(data_dir, 'alias_vocab.txt')
        self.alias_entities = [w.strip() for w in open(self.alias_path, encoding='utf8') if w.strip()]
        self.symptom_path = os.path.join(data_dir, 'symptom_vocab.txt')
        self.symptom_entities = [w.strip() for w in open(self.symptom_path, encoding='utf8') if w.strip()]
        self.complication_path = os.path.join(data_dir, 'complications_vocab.txt')
        self.complication_entities = [w.strip() for w in open(self.complication_path, encoding='utf8') if w.strip()]
        # 领域词
        self.region_words = list(set(self.disease_entities+self.alias_entities+self.symptom_entities))

        # 构造领域 AC automation
        self.disease_tree = self.build_automation(list(set(self.disease_entities)))
        self.alias_tree = self.build_automation(list(set(self.alias_entities)))
        self.symptom_tree = self.build_automation(list(set(self.symptom_entities)))
        self.complication_tree = self.build_automation(list(set(self.complication_entities)))

        (self.symptom_qwds, self.cureway_qwds, self.lasttime_qwds,
         self.cureprob_qwds, self.check_qwds, self.belong_qwds, self.disase_qwds) = self._qwds()
        pass

    def _qwds(self):
        symptom_qwds = ['什么症状', '哪些症状', '症状有哪些', '症状是什么', '什么表征',
                        '哪些表征', '表征是什么', '什么现象', '哪些现象', '现象有哪些',
                        '症候', '什么表现', '哪些表现', '表现有哪些', '什么行为',
                        '哪些行为', '行为有哪些', '什么状况', '哪些状况', '状况有哪些',
                        '现象是什么', '表现是什么', '行为是什么']  # 询问症状
        cureway_qwds = ['药', '药品', '用药', '胶囊', '口服液',
                        '炎片', '吃什么药', '用什么药', '怎么办', '买什么药',
                        '怎么治疗', '如何医治', '怎么医治', '怎么治', '怎么医',
                        '如何治', '医治方式', '疗法', '咋治', '咋办',
                        '咋治', '治疗方法']  # 询问治疗方法
        lasttime_qwds = ['周期', '多久', '多长时间', '多少时间', '几天',
                         '几年', '多少天', '多少小时', '几个小时', '多少年',
                         '多久能好', '痊愈', '康复']  # 询问治疗周期
        cureprob_qwds = ['多大概率能治好', '多大几率能治好', '治好希望大么', '几率', '几成',
                         '比例', '可能性', '能治', '可治', '可以治',
                         '可以医', '能治好吗', '可以治好吗', '会好吗', '能好吗',
                         '治愈吗']  # 询问治愈率
        check_qwds = ['检查什么', '检查项目', '哪些检查', '什么检查', '检查哪些',
                      '项目', '检测什么', '哪些检测', '检测哪些', '化验什么',
                      '哪些化验', '化验哪些', '哪些体检', '怎么查找', '如何查找',
                      '怎么检查', '如何检查', '怎么检测', '如何检测']  # 询问检查项目
        belong_qwds = ['属于什么科', '什么科', '科室', '挂什么', '挂哪个',
                       '哪个科', '哪些科']  # 询问科室
        disase_qwds = ['什么病', '啥病', '得了什么', '得了哪种', '怎么回事',
                       '咋回事', '回事', '什么情况', '什么问题', '什么毛病',
                       '啥毛病', '哪种病']  # 询问疾病
        return symptom_qwds, cureway_qwds, lasttime_qwds, cureprob_qwds, check_qwds, belong_qwds, disase_qwds

    @staticmethod
    def build_automation(word_list):
        """构造AC automation，加速过滤"""

        # AC自动机
        automation = ahocorasick.Automaton()
        # 向树中添加单词
        for index, word in enumerate(word_list):
            automation.add_word(word, (index, word))
        # 构建 自动状态机
        automation.make_automaton()
        return automation

    # 模式匹配, 得到匹配的词和类型。如疾病，疾病别名，并发症，症状
    def entity_reg(self, question):
        disease_word = [word[1][1] for word in self.disease_tree.iter(question)]
        alias_word = [word[1][1] for word in self.alias_tree.iter(question)]
        symptom_word = [word[1][1] for word in self.symptom_tree.iter(question)]
        complication_word = [word[1][1] for word in self.complication_tree.iter(question)]

        result = dict()
        if len(disease_word) > 0:
            result["Disease"] = disease_word
        if len(alias_word) > 0:
            result["Alias"] = alias_word
        if len(symptom_word) > 0:
            result["Symptom"] = symptom_word
        if len(complication_word) > 0:
            result["Complication"] = complication_word
        return result

    # 当全匹配失败时，就采用相似度计算来找相似的词
    def find_sim_words(self, question, result_dict):
        # 句子预处理
        sentence = re.sub("[{}]", re.escape(string.punctuation), question)
        sentence = re.sub("[，。‘’；：？、！【】]", " ", sentence)
        sentence = sentence.strip()

        # 单词
        words = [w.strip() for w in jieba.cut(sentence) if w.strip() not in self.stopwords and len(w.strip()) >= 2]

        scores_list = []
        for word in words:
            temp = [["Disease", self.disease_entities], ["Alias", self.alias_entities],
                    ["Symptom", self.symptom_entities], ["Complication", self.complication_entities]]
            for flag, entities in temp:
                # 计算 word 和 entities 的相似度
                scores = self.sim_cal(word, entities, flag)
                scores_list.extend(scores)
            pass
        temp1 = sorted(scores_list, key=lambda k: k[1], reverse=True)
        if temp1:
            result_dict[temp1[0][2]] = [temp1[0][0]]
        return result_dict

    # 采用DP方法计算编辑距离
    @staticmethod
    def edit_distance(s1, s2):
        m = len(s1)
        n = len(s2)
        solution = [[0 for j in range(n + 1)] for i in range(m + 1)]
        for i in range(len(s2) + 1):
            solution[0][i] = i
        for i in range(len(s1) + 1):
            solution[i][0] = i

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                solution[i][j] = solution[i-1][j-1] if s1[i-1] == s2[j-1] else (
                    1 + min(solution[i][j-1], min(solution[i-1][j], solution[i-1][j-1])))
                pass
            pass
        return solution[m][n]

    # 计算词语和字典中的词的相似度: 相同字符的个数/min(|A|,|B|) + 余弦相似度
    def sim_cal(self, word, entities, flag):
        scores = []
        for entity in entities:
            sim_num = len([w for w in word if w in entity])  # 词语和实体中词语相同的字符数

            temp = []

            # overlap score
            if sim_num != 0:
                score1 = sim_num / len(set(entity+word))
                temp.append(score1)
                pass

            # 余弦相似度分数
            try:
                score2 = self.model.similarity(word, entity)
                temp.append(score2)
            except:
                pass

            # 编辑距离分数
            score3 = 1 - self.edit_distance(word, entity) / (len(word) + len(entity))
            if score3:
                temp.append(score3)

            score = sum(temp) / len(temp)
            if score >= 0.7:
                scores.append((entity, score, flag))
            pass

        scores.sort(key=lambda k: k[1], reverse=True)
        return scores

    # 基于特征词分类
    @staticmethod
    def check_words(wds, sent):
        for wd in wds:
            if wd in sent:
                return True
        return False

    # 提取问题的TF-IDF特征
    def tfidf_features(self, text, vectorizer):
        words = [w.strip() for w in jieba.cut(text) if w.strip() and w.strip() not in self.stopwords]
        sents = [' '.join(words)]

        tfidf = vectorizer.transform(sents).toarray()
        return tfidf

    # 提取问题的关键词特征
    def other_features(self, text):
        features = [0] * 7
        for d in self.disase_qwds:
            if d in text:
                features[0] += 1

        for s in self.symptom_qwds:
            if s in text:
                features[1] += 1

        for c in self.cureway_qwds:
            if c in text:
                features[2] += 1

        for c in self.check_qwds:
            if c in text:
                features[3] += 1
        for p in self.lasttime_qwds:
            if p in text:
                features[4] += 1

        for r in self.cureprob_qwds:
            if r in text:
                features[5] += 1

        for d in self.belong_qwds:
            if d in text:
                features[6] += 1

        m = max(features)
        n = min(features)
        normed_features = features if m == n else [(i - n) / (m - n) for i in features]
        return np.array(normed_features)

    # 实体抽取主函数
    def extractor(self, question):
        # 实体抽取
        entity_result = self.entity_reg(question)
        if not entity_result:
            entity_result = self.find_sim_words(question, entity_result)
        types = [v for v in entity_result.keys()]  # 实体类型

        # 意图预测: 特征 + 预测
        tfidf_feature = self.tfidf_features(question, self.tfidf_model)
        other_feature = self.other_features(question)
        other_feature = np.reshape(other_feature, (1, other_feature.shape[0]))
        feature = np.concatenate((tfidf_feature, other_feature), axis=1)
        predicted = self.nb_model.predict(feature)
        intentions = [predicted[0]]

        # 已知疾病，查询症状
        if self.check_words(self.symptom_qwds, question) and ('Disease' in types or 'Alia' in types):
            intentions.append("query_symptom")
        # 已知疾病或症状，查询治疗方法
        if self.check_words(self.cureway_qwds, question) and \
                ('Disease' in types or 'Symptom' in types or 'Alias' in types or 'Complication' in types):
            intentions.append("query_cureway")
        # 已知疾病或症状，查询治疗周期
        if self.check_words(self.lasttime_qwds, question) and ('Disease' in types or 'Alia' in types):
            intentions.append("query_period")
        # 已知疾病，查询治愈率
        if self.check_words(self.cureprob_qwds, question) and ('Disease' in types or 'Alias' in types):
            intentions.append("query_rate")
        # 已知疾病，查询检查项目
        if self.check_words(self.check_qwds, question) and ('Disease' in types or 'Alias' in types):
            intentions.append("query_checklist")
        # 查询科室
        if self.check_words(self.belong_qwds, question) and \
                ('Disease' in types or 'Symptom' in types or 'Alias' in types or 'Complication' in types):
            intentions.append("query_department")
        # 已知症状，查询疾病
        if self.check_words(self.disase_qwds, question) and ("Symptom" in types or "Complication" in types):
            intentions.append("query_disease")
        # 若没有检测到意图，且已知疾病，则返回疾病的描述
        if not intentions and ('Disease' in types or 'Alias' in types):
            intentions.append("disease_describe")
        # 若是疾病和症状同时出现，且出现了查询疾病的特征词，则意图为查询疾病
        if self.check_words(self.disase_qwds, question) and ('Disease' in types or 'Alias' in types) \
                and ("Symptom" in types or "Complication" in types):
            intentions.append("query_disease")

        # 若没有识别出实体或意图则调用其它方法
        if not intentions or not types:
            intentions.append("QA_matching")

        entity_result["intentions"] = intentions
        return entity_result

    pass


class AnswerSearching:
    def __init__(self):
        self.graph = Graph("http://localhost:7474", username="neo4j", password="123456789")
        self.top_num = 10
        pass

    # 主要是根据不同的实体和意图构造cypher查询语句 data: {"Disease":[], "Alias":[], "Symptom":[], "Complication":[]}
    def question_parser(self, data):
        sql_list = []
        if data:
            for intent in data["intentions"]:
                sql = []
                if data.get("Disease"):
                    sql = self.transform_to_sql("Disease", data["Disease"], intent)
                elif data.get("Alias"):
                    sql = self.transform_to_sql("Alias", data["Alias"], intent)
                elif data.get("Symptom"):
                    sql = self.transform_to_sql("Symptom", data["Symptom"], intent)
                elif data.get("Complication"):
                    sql = self.transform_to_sql("Complication", data["Complication"], intent)
                    pass
                if sql:
                    sql_list.append({"intention": intent, 'sql': sql})
                pass
            pass
        return sql_list

    # 将问题转变为cypher查询语句
    @staticmethod
    def transform_to_sql(label, entities, intent):
        """
        将问题转变为cypher查询语句
        :param label:实体标签
        :param entities:实体列表
        :param intent:查询意图
        :return:cypher查询语句
        """
        if not entities:
            return []
        sql = []

        # 查询症状
        if intent == "query_symptom" and label == "Disease":
            sql = ["MATCH (d:Disease)-[:HAS_SYMPTOM]->(s) WHERE d.name='{0}' RETURN d.name,s.name".format(e)
                   for e in entities]
        if intent == "query_symptom" and label == "Alias":
            sql = ["MATCH (a:Alias)<-[:ALIAS_IS]-(d:Disease)-[:HAS_SYMPTOM]->(s) WHERE a.name='{0}' return " \
                   "d.name,s.name".format(e) for e in entities]

        # 查询治疗方法
        if intent == "query_cureway" and label == "Disease":
            sql = ["MATCH (d:Disease)-[:HAS_DRUG]->(n) WHERE d.name='{0}' return d.name,d.treatment," \
                   "n.name".format(e) for e in entities]
        if intent == "query_cureway" and label == "Alias":
            sql = ["MATCH (n)<-[:HAS_DRUG]-(d:Disease)-[]->(a:Alias) WHERE a.name='{0}' " \
                   "return d.name, d.treatment, n.name".format(e) for e in entities]
        if intent == "query_cureway" and label == "Symptom":
            sql = ["MATCH (n)<-[:HAS_DRUG]-(d:Disease)-[]->(s:Symptom) WHERE s.name='{0}' " \
                   "return d.name,d.treatment, n.name".format(e) for e in entities]
        if intent == "query_cureway" and label == "Complication":
            sql = ["MATCH (n)<-[:HAS_DRUG]-(d:Disease)-[]->(c:Complication) WHERE c.name='{0}' " \
                   "return d.name,d.treatment, n.name".format(e) for e in entities]

        # 查询治疗周期
        if intent == "query_period" and label == "Disease":
            sql = ["MATCH (d:Disease) WHERE d.name='{0}' return d.name,d.period".format(e) for e in entities]
        if intent == "query_period" and label == "Alias":
            sql = ["MATCH (d:Disease)-[]->(a:Alias) WHERE a.name='{0}' return d.name,d.period".format(e)
                   for e in entities]
        if intent == "query_period" and label == "Symptom":
            sql = ["MATCH (d:Disease)-[]->(s:Symptom) WHERE s.name='{0}' return d.name,d.period".format(e)
                   for e in entities]
        if intent == "query_period" and label == "Complication":
            sql = ["MATCH (d:Disease)-[]->(c:Complication) WHERE c.name='{0}' return d.name," \
                   "d.period".format(e) for e in entities]

        # 查询治愈率
        if intent == "query_rate" and label == "Disease":
            sql = ["MATCH (d:Disease) WHERE d.name='{0}' return d.name,d.rate".format(e) for e in entities]
        if intent == "query_rate" and label == "Alias":
            sql = ["MATCH (d:Disease)-[]->(a:Alias) WHERE a.name='{0}' return d.name,d.rate".format(e)
                   for e in entities]
        if intent == "query_rate" and label == "Symptom":
            sql = ["MATCH (d:Disease)-[]->(s:Symptom) WHERE s.name='{0}' return d.name,d.rate".format(e)
                   for e in entities]
        if intent == "query_rate" and label == "Complication":
            sql = ["MATCH (d:Disease)-[]->(c:Complication) WHERE c.name='{0}' return d.name," \
                   "d.rate".format(e) for e in entities]

        # 查询检查项目
        if intent == "query_checklist" and label == "Disease":
            sql = ["MATCH (d:Disease) WHERE d.name='{0}' return d.name,d.checklist".format(e) for e in entities]
        if intent == "query_checklist" and label == "Alias":
            sql = ["MATCH (d:Disease)-[]->(a:Alias) WHERE a.name='{0}' return d.name,d.checklist".format(e)
                   for e in entities]
        if intent == "query_checklist" and label == "Symptom":
            sql = ["MATCH (d:Disease)-[]->(s:Symptom) WHERE s.name='{0}' return d.name," \
                   "d.checklist".format(e) for e in entities]
        if intent == "query_checklist" and label == "Complication":
            sql = ["MATCH (d:Disease)-[]->(c:Complication) WHERE c.name='{0}' return d.name," \
                   "d.checklist".format(e) for e in entities]

        # 查询科室
        if intent == "query_department" and label == "Disease":
            sql = ["MATCH (d:Disease)-[:DEPARTMENT_IS]->(n) WHERE d.name='{0}' return d.name," \
                   "n.name".format(e) for e in entities]
        if intent == "query_department" and label == "Alias":
            sql = ["MATCH (n)<-[:DEPARTMENT_IS]-(d:Disease)-[:ALIAS_IS]->(a:Alias) WHERE a.name='{0}' " \
                   "return d.name,n.name".format(e) for e in entities]
        if intent == "query_department" and label == "Symptom":
            sql = ["MATCH (n)<-[:DEPARTMENT_IS]-(d:Disease)-[:HAS_SYMPTOM]->(s:Symptom) WHERE s.name='{0}' " \
                   "return d.name,n.name".format(e) for e in entities]
        if intent == "query_department" and label == "Complication":
            sql = ["MATCH (n)<-[:DEPARTMENT_IS]-(d:Disease)-[:HAS_COMPLICATION]->(c:Complication) WHERE " \
                   "c.name='{0}' return d.name,n.name".format(e) for e in entities]

        # 查询疾病
        if intent == "query_disease" and label == "Alias":
            sql = ["MATCH (d:Disease)-[]->(s:Alias) WHERE s.name='{0}' return " \
                   "d.name".format(e) for e in entities]
        if intent == "query_disease" and label == "Symptom":
            sql = ["MATCH (d:Disease)-[]->(s:Symptom) WHERE s.name='{0}' return " \
                   "d.name".format(e) for e in entities]

        # 查询疾病描述
        if intent == "disease_describe" and label == "Alias":
            sql = ["MATCH (d:Disease)-[]->(a:Alias) WHERE a.name='{0}' return d.name,d.age," \
                   "d.insurance,d.infection,d.checklist,d.period,d.rate,d.money".format(e) for e in entities]
        if intent == "disease_describe" and label == "Disease":
            sql = ["MATCH (d:Disease) WHERE d.name='{0}' return d.name,d.age,d.insurance,d.infection," \
                   "d.checklist,d.period,d.rate,d.money".format(e) for e in entities]
        if intent == "disease_describe" and label == "Symptom":
            sql = ["MATCH (d:Disease)-[]->(s:Symptom) WHERE s.name='{0}' return d.name,d.age," \
                   "d.insurance,d.infection,d.checklist,d.period,d.rate,d.money".format(e) for e in entities]
        if intent == "disease_describe" and label == "Complication":
            sql = ["MATCH (d:Disease)-[]->(c:Complication) WHERE c.name='{0}' return d.name," \
                   "d.age,d.insurance,d.infection,d.checklist,d.period,d.rate,d.money".format(e) for e in entities]

        return sql

    # 执行cypher查询，返回结果
    def searching(self, sql_list):
        final_answers = []
        for sql in sql_list:
            # 查询结果
            answers = []
            for query in sql['sql']:
                answers += self.graph.run(query).data()
                pass

            # 最终答案
            final_answer = self.answer_template(sql['intention'], answers)
            if final_answer:
                final_answers.append(final_answer)
        return final_answers

    @staticmethod
    def answer_template(intent, answers):
        """根据不同意图，返回不同模板的答案"""
        final_answer = ""
        if not answers:
            return ""

        # 查询症状
        if intent == "query_symptom":
            disease_dic = {}
            for data in answers:
                d = data['d.name']
                s = data['s.name']
                if d not in disease_dic:
                    disease_dic[d] = [s]
                else:
                    disease_dic[d].append(s)
            i = 0
            for k, v in disease_dic.items():
                if i >= 10:
                    break
                final_answer += "疾病 {0} 的症状有：{1}\n".format(k, ','.join(list(set(v))))
                i += 1
            pass

        # 查询疾病
        if intent == "query_disease":
            disease_freq = {}
            for data in answers:
                d = data["d.name"]
                disease_freq[d] = disease_freq.get(d, 0) + 1
                pass
            freq = sorted(disease_freq.items(), key=lambda x: x[1], reverse=True)
            for d, v in freq[:10]:
                final_answer += "疾病为 {0} 的概率为：{1}\n".format(d, v / 10)
                pass
            pass

        # 查询治疗方法
        if intent == "query_cureway":
            disease_dic = {}
            for data in answers:
                disease = data['d.name']
                treat = data["d.treatment"]
                drug = data["n.name"]
                if disease not in disease_dic:
                    disease_dic[disease] = [treat, drug]
                else:
                    disease_dic[disease].append(drug)
            i = 0
            for d, v in disease_dic.items():
                if i >= 10:
                    break
                final_answer += "疾病 {0} 的治疗方法有：{1}；可用药品包括：{2}\n".format(d, v[0], ','.join(v[1:]))
                i += 1
                pass
            pass

        # 查询治愈周期
        if intent == "query_period":
            disease_dic = {}
            for data in answers:
                d = data['d.name']
                p = data['d.period']
                if d not in disease_dic:
                    disease_dic[d] = [p]
                else:
                    disease_dic[d].append(p)
            i = 0
            for k, v in disease_dic.items():
                if i >= 10:
                    break
                final_answer += "疾病 {0} 的治愈周期为：{1}\n".format(k, ','.join(list(set(v))))
                i += 1
                pass
            pass

        # 查询治愈率
        if intent == "query_rate":
            disease_dic = {}
            for data in answers:
                d = data['d.name']
                r = data['d.rate']
                if d not in disease_dic:
                    disease_dic[d] = [r]
                else:
                    disease_dic[d].append(r)
            i = 0
            for k, v in disease_dic.items():
                if i >= 10:
                    break
                final_answer += "疾病 {0} 的治愈率为：{1}\n".format(k, ','.join(list(set(v))))
                i += 1
                pass
            pass

        # 查询检查项目
        if intent == "query_checklist":
            disease_dic = {}
            for data in answers:
                d = data['d.name']
                r = data['d.checklist']
                if d not in disease_dic:
                    disease_dic[d] = [r]
                else:
                    disease_dic[d].append(r)
            i = 0
            for k, v in disease_dic.items():
                if i >= 10:
                    break
                final_answer += "疾病 {0} 的检查项目有：{1}\n".format(k, ','.join(list(set(v))))
                i += 1
                pass
            pass

        # 查询科室
        if intent == "query_department":
            disease_dic = {}
            for data in answers:
                d = data['d.name']
                r = data['n.name']
                if d not in disease_dic:
                    disease_dic[d] = [r]
                else:
                    disease_dic[d].append(r)
            i = 0
            for k, v in disease_dic.items():
                if i >= 10:
                    break
                final_answer += "疾病 {0} 所属科室有：{1}\n".format(k, ','.join(list(set(v))))
                i += 1
                pass
            pass

        # 查询疾病描述
        if intent == "disease_describe":
            disease_infos = {}
            for data in answers:
                name = data['d.name']
                age = data['d.age']
                insurance = data['d.insurance']
                infection = data['d.infection']
                checklist = data['d.checklist']
                period = data['d.period']
                rate = data['d.rate']
                money = data['d.money']
                if name not in disease_infos:
                    disease_infos[name] = [age, insurance, infection, checklist, period, rate, money]
                else:
                    disease_infos[name].extend([age, insurance, infection, checklist, period, rate, money])
            i = 0
            for k, v in disease_infos.items():
                if i >= 10:
                    break
                message = "疾病 {0} 的描述信息如下：\n发病人群：{1}\n医保：{2}\n传染性：{3}\n" \
                          "检查项目：{4}\n 治愈周期：{5}\n治愈率：{6}\n费用：{7}\n"
                final_answer += message.format(k, v[0], v[1], v[2], v[3], v[4], v[5], v[6])
                i += 1
                pass
            pass

        return final_answer

    pass


class KBQA(object):

    def __init__(self):
        self.extractor = EntityExtractor()
        self.searcher = AnswerSearching()
        pass

    def qa_main(self, input_str):
        answer = "对不起，您的问题我不知道，我今后会努力改进的。"
        entities = self.extractor.extractor(input_str)  # 提取实体
        if not entities:
            return answer

        sql_list = self.searcher.question_parser(entities)  # 查询结构化
        final_answer = self.searcher.searching(sql_list)  # 获取结果
        if not final_answer:
            return answer
        else:
            return '\n'.join(final_answer)
        pass

    pass


if __name__ == "__main__":
    handler = KBQA()
    while True:
        question = input("用户：")
        if not question:
            break
        _answer = handler.qa_main(question)
        print("小豪：", _answer)
        print("*"*50)
        pass
    pass
