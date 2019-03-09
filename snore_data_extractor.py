import numpy as np
import os
from scipy import misc
from collections import Counter


class snore_data_extractor:

    def __init__(self, file_path, one_hot=False, data_mode="train", ifResize=False, resize=(224, 224), timechain=True, duplicate=True, colour_mode="RGB"):

        self.features = []
        self.labels = []
        self.dict_label_2_code = {}
        self.dict_code_2_label = {}
        self.dict_labels = {}
        self.pointer = 0
        self.one_hot = one_hot
        self.data_mode = data_mode
        if timechain == True:
            self.dict_subpart_2_timechain = {}
            self.dict_subpart_2_fullpart = {}

        # set up dictionary to translate label to code
        self.dict_label_2_code["V"] = 0
        self.dict_label_2_code["E"] = 1
        self.dict_label_2_code["O"] = 2
        self.dict_label_2_code["T"] = 3

        # set up dictionary to translate code to label
        self.dict_code_2_label[0] = "V"
        self.dict_code_2_label[1] = "E"
        self.dict_code_2_label[2] = "O"
        self.dict_code_2_label[3] = "T"

        if duplicate == True:
            self.labels_amplify = {}

        # load labels from docs
        '''self.dict_labels是一个表，表中value为train或者devel或者test的data类型(V\E\O\T)对应的数字，
        例如self.dict_labels={1:0,2:0,3:1,4:0,...,282:0}.(以snore_spectrogram_5/snore_map.txt为例)'''
        with open(file_path+"snore_map.txt", "r+") as doc:
            for _,l in enumerate(doc):
                line = l.split("\t")
                label = line[1].replace("\n", "")
                title = line[0].split("_")[0]
                number = int(line[0].split("_")[1].replace(".wav", ""))
                if data_mode == "train":
                    if title == "train":
                        self.dict_labels[number] = self.dict_label_2_code[label]
                elif data_mode == "devel":
                    if title == "devel":
                        self.dict_labels[number] = self.dict_label_2_code[label]
                elif data_mode == "test":
                    if title == "test":
                        self.dict_labels[number] = self.dict_label_2_code[label]
                        
        '''https://docs.python.org/2/library/collections.html
        duplicate==True时,即进行论文中的3.2. Reproducing Data步骤，cnt对应data中每个label对应的数字(0/1/2/3)出现了多少次，
        即cnt=counter({0：多少次，1:多少次，2：多少次，3：多少次}),0对应V，...,3对应T.
        max_value即其中出现过的最大次数,
        self.labels_amplify即为需要乘以的倍数，self.labels_amplify={0:max_value/cnt[0],...,3:max_value/cnt[3])'''
        if duplicate == True:
            cnt = Counter()
            for labels in list(self.dict_labels.values()):
                cnt[labels] += 1
            max_value = np.amax(list(cnt.values()))
            for labels in list(self.dict_code_2_label.keys()):
                label_amplify = int(np.floor(max_value/cnt[labels]))
                self.labels_amplify[labels] = label_amplify

        # load images of wavelets according to different data modes
        # during loading, resizing images can be chosen to reduce the memory allocation
        counter = 0
        for fig in os.listdir(file_path+self.data_mode):
            if timechain == True:
                [number, subnumber] = fig.replace(".png", "").split("_")
                number = int(number)
                subnumber = int(subnumber)
                figure = misc.imread(file_path+self.data_mode+"/"+fig, mode=colour_mode)
                if ifResize == True:
                    self.features.append(misc.imresize(figure, resize).tolist())
                else:
                    self.features.append(figure.tolist())
                self.labels.append(self.dict_labels[number])
                self.dict_subpart_2_timechain[counter] = subnumber
                self.dict_subpart_2_fullpart[counter] = number
            else:
                try:
                    number = fig.replace(".png", "")
                    number = int(number)
                except:
                    number = fig.replace("_.png", "")
                    number = int(number)
                figure = misc.imread(file_path+self.data_mode+"/"+fig, mode=colour_mode)
                if duplicate == True:
                    for _ in range(self.labels_amplify[self.dict_labels[number]]):
                        if ifResize == True:
                            self.features.append(misc.imresize(figure, resize).tolist())
                        else:
                            self.features.append(figure.tolist())
                        self.labels.append(self.dict_labels[number])
                else:
                    if ifResize == True:
                        self.features.append(misc.imresize(figure, resize).tolist())
                    else:
                        self.features.append(figure.tolist())
                    self.labels.append(self.dict_labels[number])
            counter += 1
            print (self.data_mode+" finish loading "+fig)

        # process labels, if one-hot is needed, then transform labels to it
        if self.one_hot == False:
            self.labels = np.array(self.labels)
        else:
            labels_temp = np.zeros([len(self.labels), 4])
            for i in range(len(self.labels)):
                labels_temp[i, self.labels[i]] = 1
            self.labels = labels_temp

    # return the full dataset
    def full_data(self):

        return self.features, self.labels

    # look up the labels by code provided
    def lookup_code(self, code):

        return self.dict_code_2_label[code]

    # look up the fullpart name by each subpart provided
    def lookup_subpart(self, seq):

        return self.dict_subpart_2_fullpart[seq]

    # look up the timechain by each subpart provided
    def lookup_timechain(self, seq):

        return self.dict_subpart_2_timechain[seq]
