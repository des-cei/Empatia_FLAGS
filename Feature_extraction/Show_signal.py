import mat73
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    data_dict = mat73.loadmat('../BBDDLab_EH_CEI_VVG_IT07.mat')
    data_list = data_dict['BBDDLab_EH_CEI_VVG_IT07']
    for index_p in range(10):
        for index_v in range(14):
            bvp_data = data_list[index_p][index_v]['EH']['Video']['raw']['gsr']
            sample_rate = 200
            # Parameters for feature
            feature_dict_all = {}
            len_frames = 3000
            overlap_frames = 2500
            # Draw the line figure

            # x = range(1000)
            # N = len(x)
            N = 1000
            fig = plt.figure(figsize=(12, 4))  # 设置画布大小
            # plt.figure()
            # plt.plot(bvp_data[3000:10000][0:-1:7], c='r')
            plt.plot(bvp_data[3000:4000], c='r')
            # plt.xticks(range(len(x)), x)  # 这里可以设置rotation参数使刻度标签旋转
            # chang x internal size
            plt.gca().margins(x=0)
            plt.gcf().canvas.draw()
            t1 = plt.gca().get_xticklabels()
            # maxsize = max([t.get_window_extent().width for t in tl])
            # 调节maxsize大小可以改变x轴放缩比例
            maxsize = 1
            m = 0.2  # inch margin
            s = maxsize / plt.gcf().dpi * N + 2 * m
            margin = m / plt.gcf().get_size_inches()[0]
            plt.gcf().subplots_adjust(left=margin, right=1. - margin)
            plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

            plt.xticks(np.linspace(0, 1000, 5))  # 设置刻度间距：表示0到1000之间分成5份
            plt.xlabel("Sample")
            plt.ylabel("Conductance (mS)")
            # plt.ylabel("Temperature (℃)")
            # plt.show()
            plt.savefig('GSR_shows/gsr_' + str(index_p) + '_' + str(index_v) + '.png', bbox_inches='tight')
            plt.close()
