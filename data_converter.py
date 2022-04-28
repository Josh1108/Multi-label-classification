'''
i/p data: List[Tuple(JD,skill)], skill list
o/p data: dict -> features_content, testid, labels_index, labels_num
'''
from asyncore import write
import csv
import pickle
import numpy
import stopwords
import json

def write_data(data, write_path):
    for row in data:
        with open(write_path,'a') as o_file:
            json.dump( row, o_file)
            o_file.write('\n')
        # write_row = json.loads(row)
        # json.dump(write_row,write_path)

def read_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data


def converter(data):
    converted_data = []
    for i,item in enumerate(data):
        dicti = dict()
        features_content = [x.lower() for x in item[0].split(' ') if x.lower() not in stopwords.get_stopwords('en')]
        testid = i
        labels_index =[i for i,x in enumerate(item[1]) if x==1]
        labels_num = len(labels_index)
        dicti['features_content'] = features_content
        dicti['testid'] = testid
        dicti['labels_index'] = labels_index
        dicti['labels_num'] = labels_num
        converted_data.append(dicti)
    return converted_data   
        



if __name__=='__main__':

    file_path = '/home/jsk/skill-prediction/data/COLING/job_dataset.valid.pkl'
    write_path ='/home/jsk/skill-prediction/Multi-Label-Text-Classification/data/job_dataset_converted_valid.json'
    data = read_data(file_path)
    data = converter(data)
    write_data(data,write_path)


