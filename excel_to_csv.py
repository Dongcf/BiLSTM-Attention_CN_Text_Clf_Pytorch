import pandas as pd
import os
from sklearn.utils import shuffle
def xlsx_to_csv_pd(excel_path,csv_path):
	data_xls = pd.read_excel(excel_path,encoding='utf-8')
	labels = data_xls['cluster_name']
	label_uni = labels.unique()
	num_class = label_uni.size
	print(num_class)
	label_map = {label: ind for ind, label in enumerate(label_uni)}
	print(label_map)
	data_xls['cluster_name'] = data_xls.cluster_name.map(label_map)
	data_xls = shuffle(data_xls)
	length = len(data_xls)
	train_xls = data_xls[:int(length*0.7)]
	dev_xls = data_xls[int(length*0.7):int(length*0.9)]
	test_xls = data_xls[int(length*0.9):]
	train_xls.to_csv(os.path.join(csv_path,"train.csv"),encoding='utf-8')
	dev_xls.to_csv(os.path.join(csv_path,"dev.csv"),encoding='utf-8')
	test_xls.to_csv(os.path.join(csv_path,'test.csv'),encoding='utf-8')
if __name__ == '__main__':
	excle_path = 'E:\项目\deep_learning_code\TextClassification\data\血管瘤_intention_0708.xlsx'
	csv_path = 'E:\项目\deep_learning_code\Attention\data'
	xlsx_to_csv_pd(excle_path,csv_path)

