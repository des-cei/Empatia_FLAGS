import numpy as np
import mat73
import math
import sys
import traceback
from biosppy.signals import eda


def cal_dicrotic(filterd_signal, peak_list):
    dicrotic_notch_index = []
    for i in range(len(peak_list) - 1):
        dicrotic_notch_index.append(peak_list[i] + np.argmin(filterd_signal[peak_list[i]:peak_list[i + 1]]))
    dicrotic_notch_index = np.array(dicrotic_notch_index)
    dicrotic_notch_value = filterd_signal[dicrotic_notch_index]
    return dicrotic_notch_index, dicrotic_notch_value


class GSR_features:
    def __init__(self, signal, sampling_rate):
        self.signal = signal
        self.sampling_rate = sampling_rate

    def GSR_mean(self):
        try:
            # Calculate GSR mean
            gsr_signal = self.signal
            feature = np.mean(gsr_signal)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def GSR_std(self):
        try:
            # Calculate GSR mean
            gsr_signal = self.signal
            feature = np.std(gsr_signal)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    # This function calculates Power Spectral Density of input signal divided in 10 Bands according to sample frequency.
    # Need to input the sample frequency from 1 to 10
    def GSR_PSD(self, Bands):
        try:
            gsr_signal = self.signal
            L = len(gsr_signal)
            Fs = self.sampling_rate
            # fft calculation of signal
            Y = np.fft.fft(gsr_signal)
            # Bilateral and Unilateral Spectral calculations
            P2 = abs(Y / (Fs * L))
            P1 = P2[: int((L / 2 + 1))]
            P1[2: -2] = 2 * P1[2: -2]
            # Number of samples of fft output to compute for each band power spectral calculation
            step = math.floor((L / 2) / Bands)
            # Frequency delimitters of each band.
            f = np.arange(0, (L // 2) + 1) * Fs / L
            # Calculate PSD
            psd = np.zeros(Bands)
            f_psd = np.zeros(Bands)
            target = step
            sample = 0
            # Loop to step through fft vector
            for i in range(Bands * step):
                # If fft value belongs to computing band, add it to psd(band) by adding it squared
                if i < target:
                    psd[sample] = psd[sample] + P1[i] ** 2
                # If fft value is last of psd(band), add it by adding it squared and get next frequency delimitter.
                elif i == target:
                    psd[sample] = psd[sample] + P1[i] ** 2
                    f_psd[sample] = f[i]
                    sample = sample + 1
                    target = i + step
            feature = psd
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    # This place is for GSR peaks_max-max to fd-pop. Ask for the meaning.
    ##
    def GSR_max(self):
        try:
            # Calculate GSR mean
            gsr_signal = self.signal
            feature = np.max(gsr_signal)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def GSR_max_number(self):
        try:
            # Calculate GSR mean
            gsr_signal = self.signal
            feature = np.sum(gsr_signal == np.max(gsr_signal))
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def GSR_min(self):
        try:
            # Calculate GSR mean
            gsr_signal = self.signal
            feature = np.min(gsr_signal)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def GSR_min_number(self):
        try:
            # Calculate GSR mean
            gsr_signal = self.signal
            feature = np.sum(gsr_signal == np.min(gsr_signal))
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def GSR_max_min_range(self):
        try:
            # Calculate GSR mean
            gsr_signal = self.signal
            feature = np.ptp(gsr_signal)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def GSR_max_min_distance(self):
        try:
            # Calculate GSR mean
            gsr_signal = self.signal
            max_index = np.where(gsr_signal == np.max(gsr_signal))
            min_index = np.where(gsr_signal == np.min(gsr_signal))
            feature = np.absolute(np.mean(max_index) - np.mean(min_index))
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def GSR_fd_mean(self):
        try:
            # Calculate GSR mean
            gsr_signal = self.signal
            dx = []
            for i in range(len(gsr_signal) - 2):
                dx.append(gsr_signal[i + 1] - gsr_signal[i])
            feature = np.mean(np.array(dx))
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def GSR_fd_std(self):
        try:
            # Calculate GSR mean
            gsr_signal = self.signal
            dx = []
            for i in range(len(gsr_signal) - 2):
                dx.append(gsr_signal[i + 1] - gsr_signal[i])
            feature = np.std(np.array(dx))
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def GSR_fd_afn(self):
        try:
            # Calculate GSR mean
            gsr_signal = self.signal
            dx = []
            for i in range(len(gsr_signal) - 2):
                dx.append(gsr_signal[i + 1] - gsr_signal[i])
            dx_array = np.array(dx)
            feature = np.mean(dx_array[np.where(dx_array < 0)])
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def GSR_fd_pon(self):
        try:
            # Calculate GSR mean
            gsr_signal = self.signal
            dx = []
            for i in range(len(gsr_signal) - 2):
                dx.append(gsr_signal[i + 1] - gsr_signal[i])
            dx_array = np.array(dx)
            feature = np.where(dx_array < 0)[0].size / dx_array.size
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def GSR_fd_afp(self):
        try:
            # Calculate GSR mean
            gsr_signal = self.signal
            dx = []
            for i in range(len(gsr_signal) - 2):
                dx.append(gsr_signal[i + 1] - gsr_signal[i])
            dx_array = np.array(dx)
            feature = np.mean(dx_array[np.where(dx_array > 0)])
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def GSR_fd_pop(self):
        try:
            # Calculate GSR mean
            gsr_signal = self.signal
            dx = []
            for i in range(len(gsr_signal) - 2):
                dx.append(gsr_signal[i + 1] - gsr_signal[i])
            dx_array = np.array(dx)
            feature = np.where(dx_array > 0)[0].size / dx_array.size
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def GSR_scl(self):
        try:
            # Get matrix size, equal to length of signal
            matrix_size = len(self.signal)
            ## Calculate D2 matrix
            # Auxiliar diagonal elements of D2 matrix
            diag_ppal = np.ones(matrix_size - 2)
            diag_2 = -2 * np.ones(matrix_size - 3)
            diag_3 = np.ones(matrix_size - 4)
            A = np.diag(diag_ppal, 0)
            B = A + np.diag(diag_2, 1)
            C = B + np.diag(diag_3, 2)
            vector1 = np.zeros(matrix_size - 2)
            # vector1[matrix_size - 3, matrix_size - 2] = [1, -2]
            vector1[matrix_size - 4] = 1
            vector1[matrix_size - 3] = -2
            vector2 = np.zeros(matrix_size - 2)
            vector2[matrix_size - 3] = 1
            D2 = np.concatenate((C, np.expand_dims(vector1, 1), np.expand_dims(vector2, 1)), axis=1)
            # Transpose D2 matrix
            D2_t = np.transpose(D2)
            ## Lambda parameter
            landa = 1500
            ## Solve problem
            Ident = np.identity(matrix_size)
            # SCL using the least square method
            SCL = np.linalg.solve(Ident + ((landa) ** 2) * np.matmul(D2_t, D2), self.signal)
            # SCR subtracting SCL to signal
            feature = SCL
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def GSR_scl_mean(self):
        try:
            SCL = self.GSR_scl()
            feature = np.mean(SCL)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def GSR_scl_std(self):
        try:
            SCL = self.GSR_scl()
            feature = np.std(SCL)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    ## Need scl-time_corr
    ##

    def GSR_scr(self):
        try:
            # Get matrix size, equal to length of signal
            matrix_size = len(self.signal)
            ## Calculate D2 matrix
            # Auxiliar diagonal elements of D2 matrix
            diag_ppal = np.ones(matrix_size - 2)
            diag_2 = -2 * np.ones(matrix_size - 3)
            diag_3 = np.ones(matrix_size - 4)
            A = np.diag(diag_ppal, 0)
            B = A + np.diag(diag_2, 1)
            C = B + np.diag(diag_3, 2)
            vector1 = np.zeros(matrix_size - 2)
            # vector1[matrix_size - 3, matrix_size - 2] = [1, -2]
            vector1[matrix_size - 4] = 1
            vector1[matrix_size - 3] = -2
            vector2 = np.zeros(matrix_size - 2)
            vector2[matrix_size - 3] = 1
            D2 = np.concatenate((C, np.expand_dims(vector1, 1), np.expand_dims(vector2, 1)), axis=1)
            # Transpose D2 matrix
            D2_t = np.transpose(D2)
            ## Lambda parameter
            landa = 1500
            ## Solve problem
            Ident = np.identity(matrix_size)
            # SCL using the least square method
            SCL = np.linalg.solve(Ident + ((landa) ** 2) * np.matmul(D2_t, D2), self.signal)
            # SCR subtracting SCL to signal
            SCR = self.signal - SCL
            feature = SCR
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def GSR_scr_mean(self):
        try:
            SCR = self.GSR_scr()
            feature = np.mean(SCR)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def GSR_scr_std(self):
        try:
            SCR = self.GSR_scr()
            feature = np.std(SCR)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def GSR_scr_max(self):
        try:
            SCR = self.GSR_scr()
            feature = np.max(SCR)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def GSR_scr_min(self):
        try:
            SCR = self.GSR_scr()
            feature = np.min(SCR)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def GSR_scr_range(self):
        try:
            # Calculate GSR mean
            feature = self.GSR_scr_max() - self.GSR_scr_min()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def GSR_scr_distance(self):
        try:
            # Calculate GSR mean
            SCR = self.GSR_scr()
            max_index = np.where(SCR == np.max(SCR))
            min_index = np.where(SCR == np.min(SCR))
            feature = np.absolute(np.mean(max_index) - np.mean(min_index))
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    ## Need scr range, distance, area

    def GSR_sd_mean(self):
        try:
            gsr_signal = self.signal
            dx = []
            for i in range(len(gsr_signal) - 2):
                dx.append(gsr_signal[i + 1] - gsr_signal[i])
            ddx = []
            for j in range(len(dx) - 2):
                ddx.append(dx[j + 1] - dx[j])
            feature = np.mean(np.array(ddx))
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def GSR_sd_std(self):
        try:
            gsr_signal = self.signal
            dx = []
            for i in range(len(gsr_signal) - 2):
                dx.append(gsr_signal[i + 1] - gsr_signal[i])
            ddx = []
            for j in range(len(dx) - 2):
                ddx.append(dx[j + 1] - dx[j])
            feature = np.std(np.array(ddx))
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def GSR_get_features_dic(self):
        features_dict = {}
        # Test GSR mean
        gsr_mean = self.GSR_mean()
        features_dict['gsr_mean'] = gsr_mean
        # Test GSR std
        gsr_std = self.GSR_std()
        features_dict['gsr_std'] = gsr_std
        
        # For GSR PSD 10
        gsr_psd10 = self.GSR_PSD(Bands=10)
        features_dict['gsr_psd1'] = gsr_psd10[0]
        features_dict['gsr_psd2'] = gsr_psd10[1]
        features_dict['gsr_psd3'] = gsr_psd10[2]
        features_dict['gsr_psd4'] = gsr_psd10[3]
        features_dict['gsr_psd5'] = gsr_psd10[4]
        features_dict['gsr_psd6'] = gsr_psd10[5]
        features_dict['gsr_psd7'] = gsr_psd10[6]
        features_dict['gsr_psd8'] = gsr_psd10[7]
        features_dict['gsr_psd9'] = gsr_psd10[8]
        features_dict['gsr_psd10'] = gsr_psd10[9]

        # Test GSR max
        gsr_max = self.GSR_max()
        features_dict['gsr_max'] = gsr_max
        # Test GSR max number
        gsr_max_number = self.GSR_max_number()
        features_dict['gsr_max_number'] = gsr_max_number
        # Test GSR min
        gsr_min = self.GSR_min()
        features_dict['gsr_min'] = gsr_min
        # Test GSR min_number
        gsr_min_number = self.GSR_min_number()
        features_dict['gsr_min_number'] = gsr_min_number
        # Test GSR max_min_range
        gsr_max_min_range = self.GSR_max_min_range()
        features_dict['gsr_max_min_range'] = gsr_max_min_range
        # Test GSR max_min_distance
        gsr_max_min_distance = self.GSR_max_min_distance()
        features_dict['gsr_max_min_distance'] = gsr_max_min_distance
        # Test GSR fd_mean
        gsr_fd_mean = self.GSR_fd_mean()
        features_dict['gsr_fd_mean'] = gsr_fd_mean
        # Test GSR fd_std
        gsr_fd_std = self.GSR_fd_std()
        features_dict['gsr_fd_std'] = gsr_fd_std
        # Test GSR fd_afn
        gsr_fd_afn = self.GSR_fd_afn()
        features_dict['gsr_fd_afn'] = gsr_fd_afn
        # Test GSR fd_pon
        gsr_fd_pon = self.GSR_fd_pon()
        features_dict['gsr_fd_pon'] = gsr_fd_pon
        # Test GSR fd_afp
        gsr_fd_afp = self.GSR_fd_afp()
        features_dict['gsr_fd_afp'] = gsr_fd_afp
        # Test GSR fd_pop
        gsr_fd_pop = self.GSR_fd_pop()
        features_dict['gsr_fd_pop'] = gsr_fd_pop

        # get scl
        SCL = self.GSR_scl()
        # Test GSR SCL mean
        gsr_scl_mean = self.GSR_scl_mean()
        features_dict['gsr_scl_mean'] = gsr_scl_mean
        # Test GSR SCL std
        gsr_scl_std = self.GSR_scl_std()
        features_dict['gsr_scl_std'] = gsr_scl_std
        # get scr
        SCR = self.GSR_scr()
        # Test GSR SCR mean
        gsr_scr_mean = self.GSR_scr_mean()
        features_dict['gsr_scr_mean'] = gsr_scr_mean
        # Test GSR SCR std
        gsr_scr_std = self.GSR_scr_std()
        features_dict['gsr_scr_std'] = gsr_scr_std
        # Test GSR SCR max
        gsr_scr_max = self.GSR_scr_max()
        features_dict['gsr_scr_max'] = gsr_scr_max
        # Test GSR SCR min
        gsr_scr_min = self.GSR_scr_min()
        features_dict['gsr_scr_min'] = gsr_scr_min
        # Test GSR SCR range
        gsr_scr_range = self.GSR_scr_range()
        features_dict['gsr_scr_range'] = gsr_scr_range
        # Test GSR SCR distance
        gsr_scr_distance = self.GSR_scr_distance()
        features_dict['gsr_scr_distance'] = gsr_scr_distance
        # Test GSR SD mean
        gsr_sd_mean = self.GSR_sd_mean()
        features_dict['gsr_sd_mean'] = gsr_sd_mean
        # Test GSR SD std
        gsr_sd_std = self.GSR_sd_std()
        features_dict['gsr_sd_std'] = gsr_sd_std
        
        return features_dict


if __name__ == '__main__':
    data_dict = mat73.loadmat('../BBDDLab_EH_CEI_VVG_IT07.mat')
    data_list = data_dict['BBDDLab_EH_CEI_VVG_IT07']
    ## sample
    raw_data_sample = data_list[0][0]['EH']['Video']['raw']['gsr_uS_filtered_sm'][:3000]
    # define signal is raw_data_sample, frequency is 200
    gsr_features = GSR_features(raw_data_sample, 200)
    features_dict = gsr_features.GSR_get_features_dic()

    for key, value in features_dict.items():
        print('{key}:{value}'.format(key=key, value=value))

    all_vlaue = {}
    for k, v in features_dict.items():
        if v in all_vlaue:
            all_vlaue[v] = all_vlaue.get(v) + 1
        else:
            all_vlaue[v] = 1
    print(all_vlaue)

    print(len(features_dict.values()) == len(set(features_dict.values())))
