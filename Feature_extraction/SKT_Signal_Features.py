import numpy as np
import mat73
import sys
import traceback


class SKT_features:

    def __init__(self, signal, sampling_rate):
        self.signal = signal
        self.sampling_rate = sampling_rate

    def SKT_mean(self):
        try:
            # Calculate SKT mean
            skt_list = self.signal
            feature = np.mean(skt_list)

        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def SKT_std(self):
        try:
            # Calculate SKT mean
            skt_list = self.signal
            feature = np.std(skt_list)

        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def SKT_pow(self):
        try:
            # Calculate SKT mean
            sqr_sum = 0
            skt_list = self.signal
            for i in range(len(skt_list)):
                sqr_sum += np.square(skt_list[i])
            SKT_pow = (1 / len(skt_list)) * sqr_sum
            feature = SKT_pow

        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def SKT_temp_t0(self):
        try:
            signal = self.signal
            Fs = self.sampling_rate
            T0 = (1 / Fs) * np.sum(signal[:Fs])  # average of the first Fs samples
            feature = T0
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def SKT_temp_t1(self):
        try:
            signal = self.signal
            Fs = self.sampling_rate
            T1 = (1 / Fs) * np.sum(signal[-Fs:])  # average of the last Fs samples
            feature = T1
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def SKT_get_features_dic(self):
        features_dict = {}
        # Test SKT mean
        skt_mean = self.SKT_mean()
        features_dict['skt_mean'] = skt_mean
        # Test SKT std
        skt_std = self.SKT_std()
        features_dict['skt_std'] = skt_std
        # Test SKT pow
        skt_pow = self.SKT_pow()
        features_dict['skt_pow'] = skt_pow
        # Test SKT temp T0
        skt_temp_t0 = self.SKT_temp_t0()
        features_dict['skt_temp_t0'] = skt_temp_t0
        # Test SKT temp T1
        skt_temp_t1 = self.SKT_temp_t1()
        features_dict['skt_temp_t1'] = skt_temp_t1

        return features_dict

if __name__ == '__main__':
    data_dict = mat73.loadmat('../BBDDLab_EH_CEI_VVG_IT07.mat')
    data_list = data_dict['BBDDLab_EH_CEI_VVG_IT07']
    ## sample
    raw_data_sample = data_list[0][0]['EH']['Video']['raw']['skt_filt_dn_sm']
    # define signal is raw_data_sample, frequency is 200
    skt_features = SKT_features(raw_data_sample, 10)
    features_dict = skt_features.SKT_get_features_dic()


    for key, value in features_dict.items():
        print('{key}:{value}'.format(key=key, value=value))
