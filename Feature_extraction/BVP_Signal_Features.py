import numpy as np
import mat73
from biosppy.signals import bvp
from scipy.signal import lombscargle
from distfit import distfit
import sys
import traceback
from Feature_extraction.SOP_del import SOP_del

# Define the functions reused from Matlab
def lomb(t, y):
    """Compute the Lomb-Scargle periodogram

    Inputs
        t : sample times
        y : measurement values

    Outputs
        f   : frequency vector
        pxx : power spectral density (PSD) estimate

    Inputs are assumed to be vectors of type numpy.ndarray.
    """
    n = t.size
    ofac = 4  # Oversampling factor
    hifac = 1
    T = t[-1] - t[0]
    Ts = T / (n - 1)
    nout = np.round(0.5 * ofac * hifac * n)
    f = np.arange(1, nout + 1) / (n * Ts * ofac)
    f_ang = f * 2 * np.pi
    pxx = lombscargle(t, y, f_ang, precenter=True)
    pxx = pxx * 2 / n
    return f, pxx

# Define the calculation of dicrotic
def cal_dicrotic(filterd_signal, peak_list):
    dicrotic_notch_index = []
    for i in range(len(peak_list) - 1):
        dicrotic_notch_index.append(peak_list[i] + np.argmin(filterd_signal[peak_list[i]:peak_list[i + 1]]))
    dicrotic_notch_index = np.array(dicrotic_notch_index)
    dicrotic_notch_value = filterd_signal[dicrotic_notch_index]
    return dicrotic_notch_index, dicrotic_notch_value


class BVP_features:

    def __init__(self, signal, slope_list, onset_list, peak_list, sampling_rate):
        self.signal = signal
        self.slope_list = slope_list
        self.onset_list = onset_list
        self.peak_list = peak_list
        self.sampling_rate = sampling_rate
        self.resp_rate = 15

    def PP_mean(self):
        try:
            # Calculate PP mean
            peak_list = self.peak_list
            PP_list = []
            for i in range(len(peak_list) - 1):
                PP_list.append(peak_list[i + 1] - peak_list[i])
            feature = np.mean(PP_list)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PP_lf_c(self):
        try:
            # Calculate PP low frequency quotient
            peak_list = self.peak_list
            PP_list = []
            t_PP = []
            for i in range(len(peak_list) - 2):
                PP_list.append(peak_list[i + 1] - peak_list[i])
                t_PP.append(peak_list[i + 1] / self.sampling_rate)
            PP_array = np.array(PP_list)
            t_PP_array = np.array(t_PP)
            [freq, PXX] = lomb(t_PP_array, PP_array)
            # Ultra Low Frequency PSD calculation:
            Ultra_tuple = np.where((freq >= 0) & (freq <= 0.04))
            Ultra_list = Ultra_tuple[0]
            ULF_PSD = 0
            for i in Ultra_list:
                pxx_now = PXX[i]
                ULF_PSD += np.square(PXX[i])
            # Low Frequency PSD calculation:
            Low_tuple = np.where((freq >= 0.04) & (freq <= 0.15))
            Low_list = Low_tuple[0]
            LF_PSD = 0
            for i in Low_list:
                pxx_now = PXX[i]
                LF_PSD += np.square(PXX[i])
            # Full PSD calculation:
            PSD = np.sum(np.square(PXX))
            # low frequency cotient feature calculation and assignation:
            feature = LF_PSD / (PSD - ULF_PSD)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PP_hf_c(self):
        try:
            # Calculate PP high frequency quotient
            peak_list = self.peak_list
            PP_list = []
            t_PP = []
            for i in range(len(peak_list) - 2):
                PP_list.append(peak_list[i + 1] - peak_list[i])
                t_PP.append(peak_list[i + 1] / self.sampling_rate)
            PP_array = np.array(PP_list)
            t_PP_array = np.array(t_PP)
            [freq, PXX] = lomb(t_PP_array, PP_array)
            # Ultra Low Frequency PSD calculation:
            Ultra_tuple = np.where((freq >= 0) & (freq <= 0.04))
            Ultra_list = Ultra_tuple[0]
            ULF_PSD = 0
            for i in Ultra_list:
                pxx_now = PXX[i]
                ULF_PSD += np.square(PXX[i])
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # Full PSD calculation:
            PSD = np.sum(np.square(PXX))
            # low frequency cotient feature calculation and assignation:
            feature = HF_PSD / (PSD - ULF_PSD)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PP_lf_hf_s(self):
        try:
            # Calculate PP low frequency high frequency sum
            peak_list = self.peak_list
            PP_list = []
            t_PP = []
            for i in range(len(peak_list) - 2):
                PP_list.append(peak_list[i + 1] - peak_list[i])
                t_PP.append(peak_list[i + 1] / self.sampling_rate)
            PP_array = np.array(PP_list)
            t_PP_array = np.array(t_PP)
            [freq, PXX] = lomb(t_PP_array, PP_array)
            # Low Frequency PSD calculation:
            Low_tuple = np.where((freq >= 0.04) & (freq <= 0.15))
            Low_list = Low_tuple[0]
            LF_PSD = 0
            for i in Low_list:
                pxx_now = PXX[i]
                LF_PSD += np.square(PXX[i])
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # low frequency - high frequency sum feature calculation and assignation:
            feature = LF_PSD + (1 / HF_PSD)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PP_lf_hf_c(self):
        try:
            # Calculate PP low frequency high frequency quotient
            peak_list = self.peak_list
            PP_list = []
            t_PP = []
            for i in range(len(peak_list) - 2):
                PP_list.append(peak_list[i + 1] - peak_list[i])
                t_PP.append(peak_list[i + 1] / self.sampling_rate)
            PP_array = np.array(PP_list)
            t_PP_array = np.array(t_PP)
            [freq, PXX] = lomb(t_PP_array, PP_array)
            # Low Frequency PSD calculation:
            Low_tuple = np.where((freq >= 0.04) & (freq <= 0.15))
            Low_list = Low_tuple[0]
            LF_PSD = 0
            for i in Low_list:
                pxx_now = PXX[i]
                LF_PSD += np.square(PXX[i])
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # low frequency - high frequency cotient feature calculation and assignation:
            feature = LF_PSD / HF_PSD
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PP_gauss(self):
        try:
            # calculates the gauss feature in a settled range of frequencies (0.15hz to 0.4 hz)
            # for a given biomarker distance vector and its time stamps.
            peak_list = self.peak_list
            PP_list = []
            t_PP = []
            for i in range(len(peak_list) - 2):
                PP_list.append(peak_list[i + 1] - peak_list[i])
                t_PP.append(peak_list[i + 1] / self.sampling_rate)
            PP_array = np.array(PP_list)
            t_PP_array = np.array(t_PP)
            [freq, PXX] = lomb(t_PP_array, PP_array)
            # Check the ammount of thata in the frequency band we desire to get the feature from
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            if len(PXX[High_list]) > 1:
                try:
                    dist = distfit(distr='norm', verbose=0)
                    Gaussian_dist = dist.fit_transform(PXX[High_list])
                    feature = np.average(Gaussian_dist['histdata'][0])
                except:
                    print('The used vector for the Gauss calc feature extraction is too short. '
                          'Try changing the window size. A null value is assigned to this feature')
                    feature = 0
            else:
                feature = 0
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PP_rr_hf_pond(self):
        try:
            # Calculate the rr high frequency ponderation feature
            peak_list = self.peak_list
            PP_list = []
            t_PP = []
            for i in range(len(peak_list) - 2):
                PP_list.append(peak_list[i + 1] - peak_list[i])
                t_PP.append(peak_list[i + 1] / self.sampling_rate)
            PP_array = np.array(PP_list)
            t_PP_array = np.array(t_PP)
            [freq, PXX] = lomb(t_PP_array, PP_array)
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # High Frequency sum calculation:
            HF_PSD_freq = sum(np.dot(HF_PSD, freq[np.where((freq >= 0.15) & (freq <= 0.4))]))
            # rr hf pond feature calculation and assignation:
            feature = HF_PSD_freq / HF_PSD
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PP_sd1_fsd1(self):
        try:
            # Calculate PP sd1
            peak_list = self.peak_list
            PP_list = []
            for i in range(len(peak_list) - 1):
                PP_list.append(peak_list[i + 1] - peak_list[i])
            sd1_vec = []
            for i in range(len(PP_list) - 2):
                sd1_vec.append(PP_list[i] - PP_list[i + 1])
            # Calculate sd1
            sd1 = np.std((np.sqrt(2) / 2) * np.array(sd1_vec))
            feature = sd1
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PP_sd1_T(self):
        try:
            # Calculate PP sd1
            feature = 4 * self.PP_sd1_fsd1()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PP_sd2_fsd2(self):
        try:
            # Calculate PP sd2
            peak_list = self.peak_list
            PP_list = []
            for i in range(len(peak_list) - 1):
                PP_list.append(peak_list[i + 1] - peak_list[i])
            sd2_vec = []
            for i in range(len(PP_list) - 2):
                sd2_vec.append(PP_list[i] + PP_list[i + 1])
            # Calculate sd2
            sd2 = np.std((np.sqrt(2) / 2) * np.array(sd2_vec))
            feature = sd2
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PP_sd2_L(self):
        try:
            feature = 4 * self.PP_sd2_fsd2()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PP_sd2_csi(self):
        try:
            feature = self.PP_sd2_L() / self.PP_sd1_T()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PP_sd2_mcsi(self):
        try:
            feature = np.square(self.PP_sd2_L()) / self.PP_sd1_T()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PP_sd2_cvi(self):
        try:
            feature = self.PP_sd2_L() * self.PP_sd1_T()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def HR_mean(self):
        try:
            # Calculate the mean of HR
            peak_list = self.peak_list
            PP_list = []
            for i in range(len(peak_list) - 1):
                PP_list.append(peak_list[i + 1] - peak_list[i])
            HR = 60.0 / (np.array(PP_list) / self.sampling_rate)
            feature = np.average(HR)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def HR_std(self):
        try:
            # Calculate the std of HR
            peak_list = self.peak_list
            PP_list = []
            for i in range(len(peak_list) - 1):
                PP_list.append(peak_list[i + 1] - peak_list[i])
            HR = 60.0 / (np.array(PP_list) / self.sampling_rate)
            feature = np.std(HR)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def HRV_mean(self):
        try:
            # Calculate the mean of HRV
            peak_list = self.peak_list
            PP_list = []
            for i in range(len(peak_list) - 1):
                PP_list.append(peak_list[i + 1] - peak_list[i])
            HR = 60.0 / (np.array(PP_list) / self.sampling_rate)
            hrv_val = np.diff(HR) / HR[:-1]
            feature = np.average(hrv_val)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def HRV_std(self):
        try:
            # Calculate the std of HRV
            peak_list = self.peak_list
            PP_list = []
            for i in range(len(peak_list) - 1):
                PP_list.append(peak_list[i + 1] - peak_list[i])
            HR = 60.0 / (np.array(PP_list) / self.sampling_rate)
            hrv_val = np.diff(HR) / HR[:-1]
            feature = np.std(hrv_val)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def HRV_rms(self):
        try:
            # Calculate the rms of HRV
            peak_list = self.peak_list
            PP_list = []
            for i in range(len(peak_list) - 1):
                PP_list.append(peak_list[i + 1] - peak_list[i])
            HR = 60.0 / (np.array(PP_list) / self.sampling_rate)
            hrv_val = np.diff(HR) / HR[:-1]
            feature = np.sqrt(np.mean(hrv_val ** 2))
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def HRV_nn50(self):
        NN50 = 0
        try:
            # Calculate the nn50 of HRV
            peak_list = self.peak_list
            PP_list = []
            for i in range(len(peak_list) - 1):
                PP_list.append(peak_list[i + 1] - peak_list[i])
            HR = 60.0 / (np.array(PP_list) / self.sampling_rate)
            hrv_val = np.diff(HR) / HR[:-1]
            for q in range(1, len(hrv_val) - 1):
                if np.abs(hrv_val[q] - hrv_val[q + 1]) * self.sampling_rate > (50 / 1000) * self.sampling_rate:
                    NN50 += 1
            feature = float(NN50)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def HRV_pnn50(self):
        NN50 = 0
        PNN50 = 0
        try:
            # Calculate the nn50 of HRV
            peak_list = self.peak_list
            PP_list = []
            for i in range(len(peak_list) - 1):
                PP_list.append(peak_list[i + 1] - peak_list[i])
            HR = 60.0 / (np.array(PP_list) / self.sampling_rate)
            hrv_val = np.diff(HR) / HR[:-1]
            for q in range(1, len(hrv_val) - 1):
                if np.abs(hrv_val[q] - hrv_val[q + 1]) * self.sampling_rate > (50 / 1000) * self.sampling_rate:
                    NN50 += 1
            PNN50 = NN50 / (len(hrv_val) - 1)
            feature = PNN50
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PW_mean(self):
        try:
            # Calculate PW mean
            onset_list = self.onset_list
            PW = []
            t_PW = []
            for i in range(len(onset_list) - 1):
                PW.append(onset_list[i + 1] - onset_list[i])
                t_PW.append(onset_list[i + 1] / self.sampling_rate)
            feature = np.mean(PW)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PW_lf_c(self):
        try:
            # Calculate PW low frequency quotient
            onset_list = self.onset_list
            PW = []
            t_PW = []
            for i in range(len(onset_list) - 1):
                PW.append(onset_list[i + 1] - onset_list[i])
                t_PW.append(onset_list[i + 1] / self.sampling_rate)
            PW_array = np.array(PW)
            t_PW_array = np.array(t_PW)
            [freq, PXX] = lomb(t_PW_array, PW_array)
            # Ultra Low Frequency PSD calculation:
            Ultra_tuple = np.where((freq >= 0) & (freq <= 0.04))
            Ultra_list = Ultra_tuple[0]
            ULF_PSD = 0
            for i in Ultra_list:
                pxx_now = PXX[i]
                ULF_PSD += np.square(PXX[i])
            # Low Frequency PSD calculation:
            Low_tuple = np.where((freq >= 0.04) & (freq <= 0.15))
            Low_list = Low_tuple[0]
            LF_PSD = 0
            for i in Low_list:
                pxx_now = PXX[i]
                LF_PSD += np.square(PXX[i])
            # Full PSD calculation:
            PSD = np.sum(np.square(PXX))
            # low frequency cotient feature calculation and assignation:
            feature = LF_PSD / (PSD - ULF_PSD)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PW_hf_c(self):
        try:
            # Calculate PW high frequency quotient
            onset_list = self.onset_list
            PW = []
            t_PW = []
            for i in range(len(onset_list) - 1):
                PW.append(onset_list[i + 1] - onset_list[i])
                t_PW.append(onset_list[i + 1] / self.sampling_rate)
            PW_array = np.array(PW)
            t_PW_array = np.array(t_PW)
            [freq, PXX] = lomb(t_PW_array, PW_array)
            # Ultra Low Frequency PSD calculation:
            Ultra_tuple = np.where((freq >= 0) & (freq <= 0.04))
            Ultra_list = Ultra_tuple[0]
            ULF_PSD = 0
            for i in Ultra_list:
                pxx_now = PXX[i]
                ULF_PSD += np.square(PXX[i])
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # Full PSD calculation:
            PSD = np.sum(np.square(PXX))
            # low frequency cotient feature calculation and assignation:
            feature = HF_PSD / (PSD - ULF_PSD)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PW_lf_hf_s(self):
        try:
            # Calculate PW low frequency high frequency sum
            onset_list = self.onset_list
            PW = []
            t_PW = []
            for i in range(len(onset_list) - 1):
                PW.append(onset_list[i + 1] - onset_list[i])
                t_PW.append(onset_list[i + 1] / self.sampling_rate)
            PW_array = np.array(PW)
            t_PW_array = np.array(t_PW)
            [freq, PXX] = lomb(t_PW_array, PW_array)
            # Low Frequency PSD calculation:
            Low_tuple = np.where((freq >= 0.04) & (freq <= 0.15))
            Low_list = Low_tuple[0]
            LF_PSD = 0
            for i in Low_list:
                pxx_now = PXX[i]
                LF_PSD += np.square(PXX[i])
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # low frequency - high frequency sum feature calculation and assignation:
            feature = LF_PSD + (1 / HF_PSD)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PW_lf_hf_c(self):
        try:
            # Calculate PW low frequency high frequency quotient
            onset_list = self.onset_list
            PW = []
            t_PW = []
            for i in range(len(onset_list) - 1):
                PW.append(onset_list[i + 1] - onset_list[i])
                t_PW.append(onset_list[i + 1] / self.sampling_rate)
            PW_array = np.array(PW)
            t_PW_array = np.array(t_PW)
            [freq, PXX] = lomb(t_PW_array, PW_array)
            # Low Frequency PSD calculation:
            Low_tuple = np.where((freq >= 0.04) & (freq <= 0.15))
            Low_list = Low_tuple[0]
            LF_PSD = 0
            for i in Low_list:
                pxx_now = PXX[i]
                LF_PSD += np.square(PXX[i])
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # low frequency - high frequency cotient feature calculation and assignation:
            feature = LF_PSD / HF_PSD
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PW_gauss(self):
        try:
            # calculates the gauss feature in a settled range of frequencies (0.15hz to 0.4 hz)
            # for a given biomarker distance vector and its time stamps.
            onset_list = self.onset_list
            PW = []
            t_PW = []
            for i in range(len(onset_list) - 1):
                PW.append(onset_list[i + 1] - onset_list[i])
                t_PW.append(onset_list[i + 1] / self.sampling_rate)
            PW_array = np.array(PW)
            t_PW_array = np.array(t_PW)
            [freq, PXX] = lomb(t_PW_array, PW_array)
            # Check the ammount of thata in the frequency band we desire to get the feature from
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            if len(PXX[High_list]) > 1:
                try:
                    dist = distfit(distr='norm', verbose=0)
                    Gaussian_dist = dist.fit_transform(PXX[High_list])
                    feature = np.average(Gaussian_dist['histdata'][0])
                except:
                    print('The used vector for the Gauss calc feature extraction is too short. '
                          'Try changing the window size. A null value is assigned to this feature')
                    feature = 0
            else:
                feature = 0
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PW_rr_hf_pond(self):
        try:
            # Calculate the rr high frequency ponderation feature
            onset_list = self.onset_list
            PW = []
            t_PW = []
            for i in range(len(onset_list) - 1):
                PW.append(onset_list[i + 1] - onset_list[i])
                t_PW.append(onset_list[i] / self.sampling_rate)
            PW_array = np.array(PW)
            t_PW_array = np.array(t_PW)
            [freq, PXX] = lomb(t_PW_array, PW_array)
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # High Frequency sum calculation:
            HF_PSD_freq = sum(np.dot(HF_PSD, freq[np.where((freq >= 0.15) & (freq <= 0.4))]))
            # rr hf pond feature calculation and assignation:
            feature = HF_PSD_freq / HF_PSD
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    # From PW-sd1-Fsd1 to PP-sd2-cvi, can not find the code in Matlab.
    # Will write them in the future.

    def PW_sd1_fsd1(self):
        try:
            # Calculate PW_sd1
            onset_list = self.onset_list
            PW_list = []
            for i in range(len(onset_list) - 1):
                PW_list.append(onset_list[i + 1] - onset_list[i])
            sd1_vec = []
            for i in range(len(PW_list) - 2):
                sd1_vec.append(PW_list[i] - PW_list[i + 1])
            # Calculate sd1
            sd1 = np.std((np.sqrt(2) / 2) * np.array(sd1_vec))
            feature = sd1
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PW_sd1_T(self):
        try:
            feature = 4 * self.PW_sd1_fsd1()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PW_sd2_fsd2(self):
        try:
            # Calculate PW_sd2
            onset_list = self.onset_list
            PW_list = []
            for i in range(len(onset_list) - 1):
                PW_list.append(onset_list[i + 1] - onset_list[i])
            sd2_vec = []
            for i in range(len(PW_list) - 2):
                sd2_vec.append(PW_list[i] + PW_list[i + 1])
            # Calculate sd2
            sd2 = np.std((np.sqrt(2) / 2) * np.array(sd2_vec))
            feature = sd2
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PW_sd2_L(self):
        try:
            feature = 4 * self.PW_sd2_fsd2()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PW_sd2_csi(self):
        try:
            feature = self.PW_sd2_L() / self.PW_sd1_T()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PW_sd2_mcsi(self):
        try:
            feature = np.square(self.PW_sd2_L()) / self.PW_sd1_T()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PW_sd2_cvi(self):
        try:
            feature = self.PW_sd2_L() * self.PW_sd1_T()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature
    # PW sd1 and sd2 end. Need to consult Rode for the meaning of Fsd, T, Fsd2, etc.

    def PDT_mean(self):
        try:
            # Calculate PDT mean
            peak_list = self.peak_list
            onset_list = self.onset_list
            PDT = []
            t_PDT = []
            for i in range(len(onset_list) - 1):
                PDT.append(onset_list[i + 1] - peak_list[i])
                t_PDT.append(onset_list[i + 1] / self.sampling_rate)
            feature = np.mean(PDT)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PDT_lf_c(self):
        try:
            # Calculate PDT low frequency quotient
            peak_list = self.peak_list
            onset_list = self.onset_list
            PDT = []
            t_PDT = []
            for i in range(len(onset_list) - 1):
                PDT.append(onset_list[i + 1] - peak_list[i])
                t_PDT.append(onset_list[i + 1] / self.sampling_rate)
            PDT_array = np.array(PDT)
            t_PDT_array = np.array(t_PDT)
            [freq, PXX] = lomb(t_PDT_array, PDT_array)
            # Ultra Low Frequency PSD calculation:
            Ultra_tuple = np.where((freq >= 0) & (freq <= 0.04))
            Ultra_list = Ultra_tuple[0]
            ULF_PSD = 0
            for i in Ultra_list:
                pxx_now = PXX[i]
                ULF_PSD += np.square(PXX[i])
            # Low Frequency PSD calculation:
            Low_tuple = np.where((freq >= 0.04) & (freq <= 0.15))
            Low_list = Low_tuple[0]
            LF_PSD = 0
            for i in Low_list:
                pxx_now = PXX[i]
                LF_PSD += np.square(PXX[i])
            # Full PSD calculation:
            PSD = np.sum(np.square(PXX))
            # low frequency cotient feature calculation and assignation:
            feature = LF_PSD / (PSD - ULF_PSD)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PDT_hf_c(self):
        try:
            # Calculate PDT high frequency quotient
            peak_list = self.peak_list
            onset_list = self.onset_list
            PDT = []
            t_PDT = []
            for i in range(len(onset_list) - 1):
                PDT.append(onset_list[i + 1] - peak_list[i])
                t_PDT.append(onset_list[i + 1] / self.sampling_rate)
            PDT_array = np.array(PDT)
            t_PDT_array = np.array(t_PDT)
            [freq, PXX] = lomb(t_PDT_array, PDT_array)
            # Ultra Low Frequency PSD calculation:
            Ultra_tuple = np.where((freq >= 0) & (freq <= 0.04))
            Ultra_list = Ultra_tuple[0]
            ULF_PSD = 0
            for i in Ultra_list:
                pxx_now = PXX[i]
                ULF_PSD += np.square(PXX[i])
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # Full PSD calculation:
            PSD = np.sum(np.square(PXX))
            # low frequency cotient feature calculation and assignation:
            feature = HF_PSD / (PSD - ULF_PSD)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PDT_lf_hf_s(self):
        try:
            # Calculate PDT low frequency high frequency sum
            peak_list = self.peak_list
            onset_list = self.onset_list
            PDT = []
            t_PDT = []
            for i in range(len(onset_list) - 1):
                PDT.append(onset_list[i + 1] - peak_list[i])
                t_PDT.append(onset_list[i + 1] / self.sampling_rate)
            PDT_array = np.array(PDT)
            t_PDT_array = np.array(t_PDT)
            [freq, PXX] = lomb(t_PDT_array, PDT_array)
            # Low Frequency PSD calculation:
            Low_tuple = np.where((freq >= 0.04) & (freq <= 0.15))
            Low_list = Low_tuple[0]
            LF_PSD = 0
            for i in Low_list:
                pxx_now = PXX[i]
                LF_PSD += np.square(PXX[i])
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # low frequency - high frequency sum feature calculation and assignation:
            feature = LF_PSD + (1 / HF_PSD)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PDT_lf_hf_c(self):
        try:
            # Calculate PDT low frequency high frequency quotient
            peak_list = self.peak_list
            onset_list = self.onset_list
            PDT = []
            t_PDT = []
            for i in range(len(onset_list) - 1):
                PDT.append(onset_list[i + 1] - peak_list[i])
                t_PDT.append(onset_list[i + 1] / self.sampling_rate)
            PDT_array = np.array(PDT)
            t_PDT_array = np.array(t_PDT)
            [freq, PXX] = lomb(t_PDT_array, PDT_array)
            # Low Frequency PSD calculation:
            Low_tuple = np.where((freq >= 0.04) & (freq <= 0.15))
            Low_list = Low_tuple[0]
            LF_PSD = 0
            for i in Low_list:
                pxx_now = PXX[i]
                LF_PSD += np.square(PXX[i])
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # low frequency - high frequency cotient feature calculation and assignation:
            feature = LF_PSD / HF_PSD
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PDT_gauss(self):
        try:
            # calculates the gauss feature in a settled range of frequencies (0.15hz to 0.4 hz)
            # for a given biomarker distance vector and its time stamps.
            peak_list = self.peak_list
            onset_list = self.onset_list
            PDT = []
            t_PDT = []
            for i in range(len(onset_list) - 1):
                PDT.append(onset_list[i + 1] - peak_list[i])
                t_PDT.append(onset_list[i + 1] / self.sampling_rate)
            PDT_array = np.array(PDT)
            t_PDT_array = np.array(t_PDT)
            [freq, PXX] = lomb(t_PDT_array, PDT_array)
            # Check the ammount of thata in the frequency band we desire to get the feature from
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            if len(PXX[High_list]) > 1:
                try:
                    dist = distfit(distr='norm', verbose=0)
                    Gaussian_dist = dist.fit_transform(PXX[High_list])
                    feature = np.average(Gaussian_dist['histdata'][0])
                except:
                    print('The used vector for the Gauss calc feature extraction is too short. '
                          'Try changing the window size. A null value is assigned to this feature')
                    feature = 0
            else:
                feature = 0
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PDT_rr_hf_pond(self):
        try:
            # Calculate the rr high frequency ponderation feature
            peak_list = self.peak_list
            onset_list = self.onset_list
            PDT = []
            t_PDT = []
            for i in range(len(onset_list) - 1):
                PDT.append(onset_list[i + 1] - peak_list[i])
                t_PDT.append(onset_list[i + 1] / self.sampling_rate)
            PDT_array = np.array(PDT)
            t_PDT_array = np.array(t_PDT)
            [freq, PXX] = lomb(t_PDT_array, PDT_array)
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # High Frequency sum calculation:
            HF_PSD_freq = sum(np.dot(HF_PSD, freq[np.where((freq >= 0.15) & (freq <= 0.4))]))
            # rr hf pond feature calculation and assignation:
            feature = HF_PSD_freq / HF_PSD
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    # From PDT-sd1-Fsd1 to PDT-sd2-cvi, can not find the code in Matlab.
    # Will write them in the future.
    def PDT_sd1_fsd1(self):
        try:
            # Calculate PDT_sd1
            peak_list = self.peak_list
            onset_list = self.onset_list
            PDT_list = []
            for i in range(len(onset_list) - 1):
                PDT_list.append(onset_list[i + 1] - peak_list[i])
            sd1_vec = []
            for i in range(len(PDT_list) - 2):
                sd1_vec.append(PDT_list[i] - PDT_list[i + 1])
            # Calculate sd1
            sd1 = np.std((np.sqrt(2) / 2) * np.array(sd1_vec))
            feature = sd1
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PDT_sd1_T(self):
        try:
            # Calculate PDT_sd1
            feature = 4 * self.PDT_sd1_fsd1()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PDT_sd2_fsd2(self):
        try:
            # Calculate PDT_sd1
            peak_list = self.peak_list
            onset_list = self.onset_list
            PDT_list = []
            for i in range(len(onset_list) - 1):
                PDT_list.append(onset_list[i + 1] - peak_list[i])
            sd2_vec = []
            for i in range(len(PDT_list) - 2):
                sd2_vec.append(PDT_list[i] + PDT_list[i + 1])
            # Calculate sd2
            sd2 = np.std((np.sqrt(2) / 2) * np.array(sd2_vec))
            feature = sd2
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PDT_sd2_L(self):
        try:
            feature = 4 * self.PDT_sd2_fsd2()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PDT_sd2_csi(self):
        try:
            feature = self.PDT_sd2_L() / self.PDT_sd1_T()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PDT_sd2_mcsi(self):
        try:
            feature = np.square(self.PDT_sd2_L()) / self.PDT_sd1_T()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PDT_sd2_cvi(self):
        try:
            feature = self.PDT_sd2_L() * self.PDT_sd1_T()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature
    # PDT sd1 and sd2 end. Need to consult Rode for the meaning of Fsd, T, Fsd2, etc.

    def PRT_mean(self):
        try:
            # Calculate PRT mean
            peak_list = self.peak_list
            onset_list = self.onset_list
            PRT = []
            t_PRT = []
            for i in range(len(peak_list) - 1):
                PRT.append(peak_list[i + 1] - onset_list[i])
                t_PRT.append(peak_list[i + 1] / self.sampling_rate)
            feature = np.mean(PRT)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PRT_lf_c(self):
        try:
            # Calculate PRT low frequency quotient
            peak_list = self.peak_list
            onset_list = self.onset_list
            PRT = []
            t_PRT = []
            for i in range(len(peak_list) - 1):
                PRT.append(peak_list[i + 1] - onset_list[i])
                t_PRT.append(peak_list[i + 1] / self.sampling_rate)
            PRT_array = np.array(PRT)
            t_PRT_array = np.array(t_PRT)
            [freq, PXX] = lomb(t_PRT_array, PRT_array)
            # Ultra Low Frequency PSD calculation:
            Ultra_tuple = np.where((freq >= 0) & (freq <= 0.04))
            Ultra_list = Ultra_tuple[0]
            ULF_PSD = 0
            for i in Ultra_list:
                pxx_now = PXX[i]
                ULF_PSD += np.square(PXX[i])
            # Low Frequency PSD calculation:
            Low_tuple = np.where((freq >= 0.04) & (freq <= 0.15))
            Low_list = Low_tuple[0]
            LF_PSD = 0
            for i in Low_list:
                pxx_now = PXX[i]
                LF_PSD += np.square(PXX[i])
            # Full PSD calculation:
            PSD = np.sum(np.square(PXX))
            # low frequency cotient feature calculation and assignation:
            feature = LF_PSD / (PSD - ULF_PSD)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PRT_hf_c(self):
        try:
            # Calculate PRT high frequency quotient
            peak_list = self.peak_list
            onset_list = self.onset_list
            PRT = []
            t_PRT = []
            for i in range(len(peak_list) - 1):
                PRT.append(peak_list[i + 1] - onset_list[i])
                t_PRT.append(peak_list[i + 1] / self.sampling_rate)
            PRT_array = np.array(PRT)
            t_PRT_array = np.array(t_PRT)
            [freq, PXX] = lomb(t_PRT_array, PRT_array)
            # Ultra Low Frequency PSD calculation:
            Ultra_tuple = np.where((freq >= 0) & (freq <= 0.04))
            Ultra_list = Ultra_tuple[0]
            ULF_PSD = 0
            for i in Ultra_list:
                pxx_now = PXX[i]
                ULF_PSD += np.square(PXX[i])
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # Full PSD calculation:
            PSD = np.sum(np.square(PXX))
            # low frequency cotient feature calculation and assignation:
            feature = HF_PSD / (PSD - ULF_PSD)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PRT_lf_hf_s(self):
        try:
            # Calculate PRT low frequency high frequency sum
            peak_list = self.peak_list
            onset_list = self.onset_list
            PRT = []
            t_PRT = []
            for i in range(len(peak_list) - 1):
                PRT.append(peak_list[i + 1] - onset_list[i])
                t_PRT.append(peak_list[i + 1] / self.sampling_rate)
            PRT_array = np.array(PRT)
            t_PRT_array = np.array(t_PRT)
            [freq, PXX] = lomb(t_PRT_array, PRT_array)
            # Low Frequency PSD calculation:
            Low_tuple = np.where((freq >= 0.04) & (freq <= 0.15))
            Low_list = Low_tuple[0]
            LF_PSD = 0
            for i in Low_list:
                pxx_now = PXX[i]
                LF_PSD += np.square(PXX[i])
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # low frequency - high frequency sum feature calculation and assignation:
            feature = LF_PSD + (1 / HF_PSD)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PRT_lf_hf_c(self):
        try:
            # Calculate PRT low frequency high frequency quotient
            peak_list = self.peak_list
            onset_list = self.onset_list
            PRT = []
            t_PRT = []
            for i in range(len(peak_list) - 1):
                PRT.append(peak_list[i + 1] - onset_list[i])
                t_PRT.append(peak_list[i + 1] / self.sampling_rate)
            PRT_array = np.array(PRT)
            t_PRT_array = np.array(t_PRT)
            [freq, PXX] = lomb(t_PRT_array, PRT_array)
            # Low Frequency PSD calculation:
            Low_tuple = np.where((freq >= 0.04) & (freq <= 0.15))
            Low_list = Low_tuple[0]
            LF_PSD = 0
            for i in Low_list:
                pxx_now = PXX[i]
                LF_PSD += np.square(PXX[i])
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # low frequency - high frequency cotient feature calculation and assignation:
            feature = LF_PSD / HF_PSD
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PRT_gauss(self):
        try:
            # calculates the gauss feature in a settled range of frequencies (0.15hz to 0.4 hz)
            # for a given biomarker distance vector and its time stamps.
            peak_list = self.peak_list
            onset_list = self.onset_list
            PRT = []
            t_PRT = []
            for i in range(len(peak_list) - 1):
                PRT.append(peak_list[i + 1] - onset_list[i])
                t_PRT.append(peak_list[i + 1] / self.sampling_rate)
            PRT_array = np.array(PRT)
            t_PRT_array = np.array(t_PRT)
            [freq, PXX] = lomb(t_PRT_array, PRT_array)
            # Check the ammount of thata in the frequency band we desire to get the feature from
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            if len(PXX[High_list]) > 1:
                try:
                    dist = distfit(distr='norm', verbose=0)
                    Gaussian_dist = dist.fit_transform(PXX[High_list])
                    feature = np.average(Gaussian_dist['histdata'][0])
                except:
                    print('The used vector for the Gauss calc feature extraction is too short. '
                          'Try changing the window size. A null value is assigned to this feature')
                    feature = 0
            else:
                feature = 0
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PRT_rr_hf_pond(self):
        try:
            # Calculate the rr high frequency ponderation feature
            peak_list = self.peak_list
            onset_list = self.onset_list
            PRT = []
            t_PRT = []
            for i in range(len(peak_list) - 1):
                PRT.append(peak_list[i + 1] - onset_list[i])
                t_PRT.append(peak_list[i + 1] / self.sampling_rate)
            PRT_array = np.array(PRT)
            t_PRT_array = np.array(t_PRT)
            [freq, PXX] = lomb(t_PRT_array, PRT_array)
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # High Frequency sum calculation:
            HF_PSD_freq = sum(np.dot(HF_PSD, freq[np.where((freq >= 0.15) & (freq <= 0.4))]))
            # rr hf pond feature calculation and assignation:
            feature = HF_PSD_freq / HF_PSD
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    # From PRT-sd1-Fsd1 to PRT-sd2-cvi, can not find the code in Matlab.
    # Will write them in the future.
    def PRT_sd1_fsd1(self):
        try:
            # Calculate PRT_sd1
            peak_list = self.peak_list
            onset_list = self.onset_list
            PRT_list = []
            for i in range(len(peak_list) - 1):
                PRT_list.append(peak_list[i + 1] - onset_list[i])
            sd1_vec = []
            for i in range(len(PRT_list) - 2):
                sd1_vec.append(PRT_list[i] - PRT_list[i + 1])
            # Calculate sd1
            sd1 = np.std((np.sqrt(2) / 2) * np.array(sd1_vec))
            feature = sd1
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PRT_sd1_T(self):
        try:
            # Calculate PDT_sd1
            feature = 4 * self.PRT_sd1_fsd1()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PRT_sd2_fsd2(self):
        try:
            # Calculate PRT_sd1
            peak_list = self.peak_list
            onset_list = self.onset_list
            PRT_list = []
            for i in range(len(peak_list) - 1):
                PRT_list.append(peak_list[i + 1] - onset_list[i])
            sd2_vec = []
            for i in range(len(PRT_list) - 2):
                sd2_vec.append(PRT_list[i] + PRT_list[i + 1])
            # Calculate sd2
            sd2 = np.std((np.sqrt(2) / 2) * np.array(sd2_vec))
            feature = sd2
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PRT_sd2_L(self):
        try:
            feature = 4 * self.PRT_sd2_fsd2()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PRT_sd2_csi(self):
        try:
            feature = self.PRT_sd2_L() / self.PRT_sd1_T()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PRT_sd2_mcsi(self):
        try:
            feature = np.square(self.PRT_sd2_L()) / self.PRT_sd1_T()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PRT_sd2_cvi(self):
        try:
            feature = self.PRT_sd2_L() * self.PRT_sd1_T()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature
    # PRT sd1 and sd2 end. Need to consult Rode for the meaning of Fsd, T, Fsd2, etc.

    def PA_mean(self):
        try:
            # Calculate PA mean
            peak_list = self.peak_list
            peak_val = []
            for i in range(len(peak_list) - 1):
                peak_val.append(self.signal[peak_list[i]])
            onset_list = self.onset_list
            onset_val = []
            for i in range(len(onset_list) - 1):
                onset_val.append(self.signal[onset_list[i]])
            PA = []
            t_PA = []
            for i in range(len(onset_val) - 2):
                PA.append(peak_val[i + 1] - onset_val[i])
                t_PA.append(peak_list[i + 1] / self.sampling_rate)
            feature = np.mean(PA)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PA_lf_c(self):
        try:
            # Calculate PA low frequency quotient
            peak_list = self.peak_list
            peak_val = []
            for i in range(len(peak_list) - 1):
                peak_val.append(self.signal[peak_list[i]])
            onset_list = self.onset_list
            onset_val = []
            for i in range(len(onset_list) - 1):
                onset_val.append(self.signal[onset_list[i]])
            PA = []
            t_PA = []
            for i in range(len(onset_val) - 2):
                PA.append(peak_val[i + 1] - onset_val[i])
                t_PA.append(peak_list[i + 1] / self.sampling_rate)
            PA_array = np.array(PA)
            t_PA_array = np.array(t_PA)
            [freq, PXX] = lomb(t_PA_array, PA_array)
            # Ultra Low Frequency PSD calculation:
            Ultra_tuple = np.where((freq >= 0) & (freq <= 0.04))
            Ultra_list = Ultra_tuple[0]
            ULF_PSD = 0
            for i in Ultra_list:
                pxx_now = PXX[i]
                ULF_PSD += np.square(PXX[i])
            # Low Frequency PSD calculation:
            Low_tuple = np.where((freq >= 0.04) & (freq <= 0.15))
            Low_list = Low_tuple[0]
            LF_PSD = 0
            for i in Low_list:
                pxx_now = PXX[i]
                LF_PSD += np.square(PXX[i])
            # Full PSD calculation:
            PSD = np.sum(np.square(PXX))
            # low frequency cotient feature calculation and assignation:
            feature = LF_PSD / (PSD - ULF_PSD)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PA_hf_c(self):
        try:
            # Calculate PA high frequency quotient
            peak_list = self.peak_list
            peak_val = []
            for i in range(len(peak_list) - 1):
                peak_val.append(self.signal[peak_list[i]])
            onset_list = self.onset_list
            onset_val = []
            for i in range(len(onset_list) - 1):
                onset_val.append(self.signal[onset_list[i]])
            PA = []
            t_PA = []
            for i in range(len(onset_val) - 2):
                PA.append(peak_val[i + 1] - onset_val[i])
                t_PA.append(peak_list[i + 1] / self.sampling_rate)
            PA_array = np.array(PA)
            t_PA_array = np.array(t_PA)
            [freq, PXX] = lomb(t_PA_array, PA_array)
            # Ultra Low Frequency PSD calculation:
            Ultra_tuple = np.where((freq >= 0) & (freq <= 0.04))
            Ultra_list = Ultra_tuple[0]
            ULF_PSD = 0
            for i in Ultra_list:
                pxx_now = PXX[i]
                ULF_PSD += np.square(PXX[i])
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # Full PSD calculation:
            PSD = np.sum(np.square(PXX))
            # low frequency cotient feature calculation and assignation:
            feature = HF_PSD / (PSD - ULF_PSD)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PA_lf_hf_s(self):
        try:
            # Calculate PA low frequency high frequency sum
            peak_list = self.peak_list
            peak_val = []
            for i in range(len(peak_list) - 1):
                peak_val.append(self.signal[peak_list[i]])
            onset_list = self.onset_list
            onset_val = []
            for i in range(len(onset_list) - 1):
                onset_val.append(self.signal[onset_list[i]])
            PA = []
            t_PA = []
            for i in range(len(onset_val) - 2):
                PA.append(peak_val[i + 1] - onset_val[i])
                t_PA.append(peak_list[i + 1] / self.sampling_rate)
            PA_array = np.array(PA)
            t_PA_array = np.array(t_PA)
            [freq, PXX] = lomb(t_PA_array, PA_array)
            # Low Frequency PSD calculation:
            Low_tuple = np.where((freq >= 0.04) & (freq <= 0.15))
            Low_list = Low_tuple[0]
            LF_PSD = 0
            for i in Low_list:
                pxx_now = PXX[i]
                LF_PSD += np.square(PXX[i])
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # low frequency - high frequency sum feature calculation and assignation:
            feature = LF_PSD + (1 / HF_PSD)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PA_lf_hf_c(self):
        try:
            # Calculate PA low frequency high frequency quotient
            peak_list = self.peak_list
            peak_val = []
            for i in range(len(peak_list) - 1):
                peak_val.append(self.signal[peak_list[i]])
            onset_list = self.onset_list
            onset_val = []
            for i in range(len(onset_list) - 1):
                onset_val.append(self.signal[onset_list[i]])
            PA = []
            t_PA = []
            for i in range(len(onset_val) - 2):
                PA.append(peak_val[i + 1] - onset_val[i])
                t_PA.append(peak_list[i + 1] / self.sampling_rate)
            PA_array = np.array(PA)
            t_PA_array = np.array(t_PA)
            [freq, PXX] = lomb(t_PA_array, PA_array)
            # Low Frequency PSD calculation:
            Low_tuple = np.where((freq >= 0.04) & (freq <= 0.15))
            Low_list = Low_tuple[0]
            LF_PSD = 0
            for i in Low_list:
                pxx_now = PXX[i]
                LF_PSD += np.square(PXX[i])
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # low frequency - high frequency cotient feature calculation and assignation:
            feature = LF_PSD / HF_PSD
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PA_gauss(self):
        try:
            # calculates the gauss feature in a settled range of frequencies (0.15hz to 0.4 hz)
            # for a given biomarker distance vector and its time stamps.
            peak_list = self.peak_list
            peak_val = []
            for i in range(len(peak_list) - 1):
                peak_val.append(self.signal[peak_list[i]])
            onset_list = self.onset_list
            onset_val = []
            for i in range(len(onset_list) - 1):
                onset_val.append(self.signal[onset_list[i]])
            PA = []
            t_PA = []
            for i in range(len(onset_val) - 2):
                PA.append(peak_val[i + 1] - onset_val[i])
                t_PA.append(peak_list[i + 1] / self.sampling_rate)
            PA_array = np.array(PA)
            t_PA_array = np.array(t_PA)
            [freq, PXX] = lomb(t_PA_array, PA_array)
            # Check the ammount of thata in the frequency band we desire to get the feature from
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            if len(PXX[High_list]) > 1:
                try:
                    dist = distfit(distr='norm', verbose=0)
                    Gaussian_dist = dist.fit_transform(PXX[High_list])
                    feature = np.average(Gaussian_dist['histdata'][0])
                except:
                    print('The used vector for the Gauss calc feature extraction is too short. '
                          'Try changing the window size. A null value is assigned to this feature')
                    feature = 0
            else:
                feature = 0
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PA_rr_hf_pond(self):
        try:
            # Calculate the rr high frequency ponderation feature
            peak_list = self.peak_list
            peak_val = []
            for i in range(len(peak_list) - 1):
                peak_val.append(self.signal[peak_list[i]])
            onset_list = self.onset_list
            onset_val = []
            for i in range(len(onset_list) - 1):
                onset_val.append(self.signal[onset_list[i]])
            PA = []
            t_PA = []
            for i in range(len(onset_val) - 2):
                PA.append(peak_val[i + 1] - onset_val[i])
                t_PA.append(peak_list[i + 1] / self.sampling_rate)
            PA_array = np.array(PA)
            t_PA_array = np.array(t_PA)
            [freq, PXX] = lomb(t_PA_array, PA_array)
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # High Frequency sum calculation:
            HF_PSD_freq = sum(np.dot(HF_PSD, freq[np.where((freq >= 0.15) & (freq <= 0.4))]))
            # rr hf pond feature calculation and assignation:
            feature = HF_PSD_freq / HF_PSD
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PWr_mean(self):
        try:
            # Calculate PWr mean
            # Calculate Dicrotic
            peak_list = self.peak_list
            onset_list = self.onset_list
            Dicrotic_index,  Dicrotic_val = cal_dicrotic(self.signal, np.array(peak_list))
            PWr = []
            t_PWr = []
            for i in range(len(onset_list) - 2):
                PWr.append(Dicrotic_index[i] - onset_list[i])
                t_PWr.append(Dicrotic_index[i] / self.sampling_rate)
            feature = np.mean(PWr)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PWr_lf_c(self):
        try:
            # Calculate PWr low frequency quotient
            peak_list = self.peak_list
            onset_list = self.onset_list
            Dicrotic_index,  Dicrotic_val = cal_dicrotic(self.signal, np.array(peak_list))
            PWr = []
            t_PWr = []
            for i in range(len(onset_list) - 2):
                PWr.append(Dicrotic_index[i] - onset_list[i])
                t_PWr.append(Dicrotic_index[i] / self.sampling_rate)
            PWr_array = np.array(PWr)
            t_PWr_array = np.array(t_PWr)
            [freq, PXX] = lomb(t_PWr_array, PWr_array)
            # Ultra Low Frequency PSD calculation:
            Ultra_tuple = np.where((freq >= 0) & (freq <= 0.04))
            Ultra_list = Ultra_tuple[0]
            ULF_PSD = 0
            for i in Ultra_list:
                pxx_now = PXX[i]
                ULF_PSD += np.square(PXX[i])
            # Low Frequency PSD calculation:
            Low_tuple = np.where((freq >= 0.04) & (freq <= 0.15))
            Low_list = Low_tuple[0]
            LF_PSD = 0
            for i in Low_list:
                pxx_now = PXX[i]
                LF_PSD += np.square(PXX[i])
            # Full PSD calculation:
            PSD = np.sum(np.square(PXX))
            # low frequency cotient feature calculation and assignation:
            feature = LF_PSD / (PSD - ULF_PSD)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PWr_hf_c(self):
        try:
            # Calculate PWr high frequency quotient
            peak_list = self.peak_list
            onset_list = self.onset_list
            Dicrotic_index,  Dicrotic_val = cal_dicrotic(self.signal, np.array(peak_list))
            PWr = []
            t_PWr = []
            for i in range(len(onset_list) - 2):
                PWr.append(Dicrotic_index[i] - onset_list[i])
                t_PWr.append(Dicrotic_index[i] / self.sampling_rate)
            PWr_array = np.array(PWr)
            t_PWr_array = np.array(t_PWr)
            [freq, PXX] = lomb(t_PWr_array, PWr_array)
            # Ultra Low Frequency PSD calculation:
            Ultra_tuple = np.where((freq >= 0) & (freq <= 0.04))
            Ultra_list = Ultra_tuple[0]
            ULF_PSD = 0
            for i in Ultra_list:
                pxx_now = PXX[i]
                ULF_PSD += np.square(PXX[i])
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # Full PSD calculation:
            PSD = np.sum(np.square(PXX))
            # low frequency cotient feature calculation and assignation:
            feature = HF_PSD / (PSD - ULF_PSD)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PWr_lf_hf_s(self):
        try:
            # Calculate PWr low frequency high frequency sum
            peak_list = self.peak_list
            onset_list = self.onset_list
            Dicrotic_index,  Dicrotic_val = cal_dicrotic(self.signal, np.array(peak_list))
            PWr = []
            t_PWr = []
            for i in range(len(onset_list) - 2):
                PWr.append(Dicrotic_index[i] - onset_list[i])
                t_PWr.append(Dicrotic_index[i] / self.sampling_rate)
            PWr_array = np.array(PWr)
            t_PWr_array = np.array(t_PWr)
            [freq, PXX] = lomb(t_PWr_array, PWr_array)
            # Low Frequency PSD calculation:
            Low_tuple = np.where((freq >= 0.04) & (freq <= 0.15))
            Low_list = Low_tuple[0]
            LF_PSD = 0
            for i in Low_list:
                pxx_now = PXX[i]
                LF_PSD += np.square(PXX[i])
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # low frequency - high frequency sum feature calculation and assignation:
            feature = LF_PSD + (1 / HF_PSD)
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PWr_lf_hf_c(self):
        try:
            # Calculate PWr low frequency high frequency quotient
            peak_list = self.peak_list
            onset_list = self.onset_list
            Dicrotic_index,  Dicrotic_val = cal_dicrotic(self.signal, np.array(peak_list))
            PWr = []
            t_PWr = []
            for i in range(len(onset_list) - 2):
                PWr.append(Dicrotic_index[i] - onset_list[i])
                t_PWr.append(Dicrotic_index[i] / self.sampling_rate)
            PWr_array = np.array(PWr)
            t_PWr_array = np.array(t_PWr)
            [freq, PXX] = lomb(t_PWr_array, PWr_array)
            # Low Frequency PSD calculation:
            Low_tuple = np.where((freq >= 0.04) & (freq <= 0.15))
            Low_list = Low_tuple[0]
            LF_PSD = 0
            for i in Low_list:
                pxx_now = PXX[i]
                LF_PSD += np.square(PXX[i])
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # low frequency - high frequency cotient feature calculation and assignation:
            feature = LF_PSD / HF_PSD
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PWr_gauss(self):
        try:
            # calculates the gauss feature in a settled range of frequencies (0.15hz to 0.4 hz)
            # for a given biomarker distance vector and its time stamps.
            peak_list = self.peak_list
            onset_list = self.onset_list
            Dicrotic_index,  Dicrotic_val = cal_dicrotic(self.signal, np.array(peak_list))
            PWr = []
            t_PWr = []
            for i in range(len(onset_list) - 2):
                PWr.append(Dicrotic_index[i] - onset_list[i])
                t_PWr.append(Dicrotic_index[i] / self.sampling_rate)
            PWr_array = np.array(PWr)
            t_PWr_array = np.array(t_PWr)
            [freq, PXX] = lomb(t_PWr_array, PWr_array)
            # Check the ammount of thata in the frequency band we desire to get the feature from
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            if len(PXX[High_list]) > 1:
                try:
                    dist = distfit(distr='norm', verbose=0)
                    Gaussian_dist = dist.fit_transform(PXX[High_list])
                    feature = np.average(Gaussian_dist['histdata'][0])
                except:
                    print('The used vector for the Gauss calc feature extraction is too short. '
                          'Try changing the window size. A null value is assigned to this feature')
                    feature = 0
            else:
                feature = 0
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PWr_rr_hf_pond(self):
        try:
            # Calculate the rr high frequency ponderation feature
            peak_list = self.peak_list
            onset_list = self.onset_list
            Dicrotic_index,  Dicrotic_val = cal_dicrotic(self.signal, np.array(peak_list))
            PWr = []
            t_PWr = []
            for i in range(len(onset_list) - 2):
                PWr.append(Dicrotic_index[i] - onset_list[i])
                t_PWr.append(Dicrotic_index[i] / self.sampling_rate)
            PWr_array = np.array(PWr)
            t_PWr_array = np.array(t_PWr)
            [freq, PXX] = lomb(t_PWr_array, PWr_array)
            # High Frequency PSD calculation:
            High_tuple = np.where((freq >= 0.15) & (freq <= 0.4))
            High_list = High_tuple[0]
            HF_PSD = 0
            for i in High_list:
                pxx_now = PXX[i]
                HF_PSD += np.square(PXX[i])
            # High Frequency sum calculation:
            HF_PSD_freq = sum(np.dot(HF_PSD, freq[np.where((freq >= 0.15) & (freq <= 0.4))]))
            # rr hf pond feature calculation and assignation:
            feature = HF_PSD_freq / HF_PSD
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    # From PWr-sd1-Fsd1 to PWr-sd2-cvi, can not find the code in Matlab.
    # Will write them in the future.
    def PWr_sd1_fsd1(self):
        try:
            # Calculate PWr_sd1
            peak_list = self.peak_list
            onset_list = self.onset_list
            Dicrotic_index,  Dicrotic_val = cal_dicrotic(self.signal, np.array(peak_list))
            PWr_list = []
            for i in range(len(peak_list) - 1):
                PWr_list.append(Dicrotic_index[i] - onset_list[i])
            sd1_vec = []
            for i in range(len(PWr_list) - 2):
                sd1_vec.append(PWr_list[i] - PWr_list[i + 1])
            # Calculate sd1
            sd1 = np.std((np.sqrt(2) / 2) * np.array(sd1_vec))
            feature = sd1
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PWr_sd1_T(self):
        try:
            # Calculate PDT_sd1
            feature = 4 * self.PWr_sd1_fsd1()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PWr_sd2_fsd2(self):
        try:
            # Calculate PWr_sd1
            peak_list = self.peak_list
            onset_list = self.onset_list
            Dicrotic_index,  Dicrotic_val = cal_dicrotic(self.signal, np.array(peak_list))
            PWr_list = []
            for i in range(len(peak_list) - 1):
                PWr_list.append(Dicrotic_index[i] - onset_list[i])
            sd2_vec = []
            for i in range(len(PWr_list) - 2):
                sd2_vec.append(PWr_list[i] + PWr_list[i + 1])
            # Calculate sd2
            sd2 = np.std((np.sqrt(2) / 2) * np.array(sd2_vec))
            feature = sd2
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PWr_sd2_L(self):
        try:
            feature = 4 * self.PWr_sd2_fsd2()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PWr_sd2_csi(self):
        try:
            feature = self.PWr_sd2_L() / self.PWr_sd1_T()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PWr_sd2_mcsi(self):
        try:
            feature = np.square(self.PWr_sd2_L()) / self.PWr_sd1_T()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature

    def PWr_sd2_cvi(self):
        try:
            feature = self.PWr_sd2_L() * self.PWr_sd1_T()
        except Exception as e:
            print(e.args)
            print('========================')
            print(traceback.format_exc())
            print("Can not calculate feature :" + str(sys._getframe().f_code.co_name))
            feature = 0
        return feature
    # PWr sd1 and sd2 end. Need to consult Rode for the meaning of Fsd, T, Fsd2, etc.

    # Return a dictionary contained all the features.
    def BVP_get_features_dic(self):
        features_dict = {}
        # Test PP mean
        pp_mean = self.PP_mean()
        features_dict['bvp_pp_mean'] = pp_mean
        # Test PP low frequency quotient
        pp_lf_c = self.PP_lf_c()
        features_dict['bvp_pp_lf_c'] = pp_lf_c
        # Test PP hign frequency quotient
        pp_hf_c = self.PP_hf_c()
        features_dict['bvp_pp_hf_c'] = pp_hf_c
        # Test PP low frequency high frequency sum
        pp_lf_hf_s = self.PP_lf_hf_s()
        features_dict['bvp_pp_lf_hf_s'] = pp_lf_hf_s
        # Test PP low frequency high frequency quotient
        pp_lf_hf_c = self.PP_lf_hf_c()
        features_dict['bvp_pp_lf_hf_c'] = pp_lf_hf_c
        # Test PP gauss feature
        pp_gauss = self.PP_gauss()
        features_dict['bvp_pp_gauss'] = pp_gauss
        # Test rr high frequency ponderation
        pp_rr_hf_pond = self.PP_rr_hf_pond()
        features_dict['bvp_pp_rr_hf_pond'] = pp_rr_hf_pond

        # This place for PP sd calculations
        # Test PP sd1 feature
        pp_sd1_fsd1 = self.PP_sd1_fsd1()
        features_dict['bvp_pp_sd1_fsd1'] = pp_sd1_fsd1
        # Test PP sd1 T
        pp_sd1_T = self.PP_sd1_T()
        features_dict['bvp_pp_sd1_T'] = pp_sd1_T
        # Test PP sd2 feature
        pp_sd2_fsd2 = self.PP_sd2_fsd2()
        features_dict['bvp_pp_sd2_fsd2'] = pp_sd2_fsd2
        # Test PP sd2 L
        pp_sd2_L = self.PP_sd2_L()
        features_dict['bvp_pp_sd2_L'] = pp_sd2_L
        # Test PP sd2 csi feature
        pp_sd2_csi = self.PP_sd2_csi()
        features_dict['bvp_pp_sd2_csi'] = pp_sd2_csi
        # Test PP sd2 mcsi feature
        pp_sd2_mcsi = self.PP_sd2_mcsi()
        features_dict['bvp_pp_sd2_mcsi'] = pp_sd2_mcsi
        # Test PP sd2 cvi feature
        pp_sd2_cvi = self.PP_sd2_cvi()
        features_dict['bvp_pp_sd2_cvi'] = pp_sd2_cvi
        # PP sd calculations done

        # Test HR mean
        hr_mean = self.HR_mean()
        features_dict['bvp_hr_mean'] = hr_mean
        # Test HR std
        hr_std = self.HR_std()
        features_dict['bvp_hr_std'] = hr_std
        # Test HRV mean
        hrv_mean = self.HRV_mean()
        features_dict['bvp_hrv_mean'] = hrv_mean
        # Test HRV std
        hrv_std = self.HRV_std()
        features_dict['bvp_hrv_std'] = hrv_std
        # Test HRV rms
        hrv_rms = self.HRV_rms()
        features_dict['bvp_hrv_rms'] = hrv_rms
        # Test HRV nn50
        hrv_nn50 = self.HRV_nn50()
        features_dict['bvp_hrv_nn50'] = hrv_nn50
        # Test HRV pnn50
        hrv_pnn50 = self.HRV_pnn50()
        features_dict['bvp_hrv_pnn50'] = hrv_pnn50


        # Test PW mean
        pw_mean = self.PW_mean()
        features_dict['bvp_pw_mean'] = pw_mean
        # Test PW low frequency quotient
        pw_lf_c = self.PW_lf_c()
        features_dict['bvp_pw_lf_c'] = pw_lf_c
        # Test PW hign frequency quotient
        pw_hf_c = self.PW_hf_c()
        features_dict['bvp_pw_hf_c'] = pw_hf_c
        # Test PW low frequency high frequency sum
        pw_lf_hf_s = self.PW_lf_hf_s()
        features_dict['bvp_pw_lf_hf_s'] = pw_lf_hf_s
        # Test PW low frequency high frequency quotient
        pw_lf_hf_c = self.PW_lf_hf_c()
        features_dict['bvp_pw_lf_hf_c'] = pw_lf_hf_c
        # Test PW gauss feature
        pw_gauss = self.PW_gauss()
        features_dict['bvp_pw_gauss'] = pw_gauss
        # Test PW rr high frequency ponderation
        pw_rr_hf_pond = self.PW_rr_hf_pond()
        features_dict['bvp_pw_rr_hf_pond'] = pw_rr_hf_pond

        # This place for PW sd calculations
        # Test PW sd1 feature
        pw_sd1_fsd1 = self.PW_sd1_fsd1()
        features_dict['bvp_pw_sd1_fsd1'] = pw_sd1_fsd1
        # Test PW sd1 T
        pw_sd1_T = self.PW_sd1_T()
        features_dict['bvp_pw_sd1_T'] = pw_sd1_T
        # Test PW sd2 feature
        pw_sd2_fsd2 = self.PW_sd2_fsd2()
        features_dict['bvp_pw_sd2_fsd2'] = pw_sd2_fsd2
        # Test PW sd2 L
        pw_sd2_L = self.PW_sd2_L()
        features_dict['bvp_pw_sd2_L'] = pw_sd2_L
        # Test PW sd2 csi feature
        pw_sd2_csi = self.PW_sd2_csi()
        features_dict['bvp_pw_sd2_csi'] = pw_sd2_csi
        # Test PW sd2 mcsi feature
        pw_sd2_mcsi = self.PW_sd2_mcsi()
        features_dict['bvp_pw_sd2_mcsi'] = pw_sd2_mcsi
        # Test PW sd2 cvi feature
        pw_sd2_cvi = self.PW_sd2_cvi()
        features_dict['bvp_pw_sd2_cvi'] = pw_sd2_cvi
        # PW sd calculations done

        # Test PDT mean
        pdt_mean = self.PDT_mean()
        features_dict['bvp_pdt_mean'] = pdt_mean
        # Test PDT low frequency quotient
        pdt_lf_c = self.PDT_lf_c()
        features_dict['bvp_pdt_lf_c'] = pdt_lf_c
        # Test PDT hign frequency quotient
        pdt_hf_c = self.PDT_hf_c()
        features_dict['bvp_pdt_hf_c'] = pdt_hf_c
        # Test PDT low frequency high frequency sum
        pdt_lf_hf_s = self.PDT_lf_hf_s()
        features_dict['bvp_pdt_lf_hf_s'] = pdt_lf_hf_s
        # Test PDT low frequency high frequency quotient
        pdt_lf_hf_c = self.PDT_lf_hf_c()
        features_dict['bvp_pdt_lf_hf_c'] = pdt_lf_hf_c
        # Test PDT gauss feature
        pdt_gauss = self.PDT_gauss()
        features_dict['bvp_pdt_gauss'] = pdt_gauss
        # Test PDT rr high frequency ponderation
        pdt_rr_hf_pond = self.PDT_rr_hf_pond()
        features_dict['bvp_pdt_rr_hf_pond'] = pdt_rr_hf_pond

        # This place for PDT sd calculations
        # Test PDT sd1 feature
        pdt_sd1_fsd1 = self.PDT_sd1_fsd1()
        features_dict['bvp_pdt_sd1_fsd1'] = pdt_sd1_fsd1
        # Test PDT sd1 T
        pdt_sd1_T = self.PDT_sd1_T()
        features_dict['bvp_pdt_sd1_T'] = pdt_sd1_T
        # Test PDT sd2 feature
        pdt_sd2_fsd2 = self.PDT_sd2_fsd2()
        features_dict['bvp_pdt_sd2_fsd2'] = pdt_sd2_fsd2
        # Test PDT sd2 L
        pdt_sd2_L = self.PDT_sd2_L()
        features_dict['bvp_pdt_sd2_L'] = pdt_sd2_L
        # Test PDT sd2 csi feature
        pdt_sd2_csi = self.PDT_sd2_csi()
        features_dict['bvp_pdt_sd2_csi'] = pdt_sd2_csi
        # Test PDT sd2 mcsi feature
        pdt_sd2_mcsi = self.PDT_sd2_mcsi()
        features_dict['bvp_pdt_sd2_mcsi'] = pdt_sd2_mcsi
        # Test PDT sd2 cvi feature
        pdt_sd2_cvi = self.PDT_sd2_cvi()
        features_dict['bvp_pdt_sd2_cvi'] = pdt_sd2_cvi
        # PDT sd calculations done

        # Test PRT mean
        prt_mean = self.PRT_mean()
        features_dict['bvp_prt_mean'] = prt_mean
        # Test PRT low frequency quotient
        prt_lf_c = self.PRT_lf_c()
        features_dict['bvp_prt_lf_c'] = prt_lf_c
        # Test PRT hign frequency quotient
        prt_hf_c = self.PRT_hf_c()
        features_dict['bvp_prt_hf_c'] = prt_hf_c
        # Test PRT low frequency high frequency sum
        prt_lf_hf_s = self.PRT_lf_hf_s()
        features_dict['bvp_prt_lf_hf_s'] = prt_lf_hf_s
        # Test PRT low frequency high frequency quotient
        prt_lf_hf_c = self.PRT_lf_hf_c()
        features_dict['bvp_prt_lf_hf_c'] = prt_lf_hf_c
        # Test PRT gauss feature
        prt_gauss = self.PRT_gauss()
        features_dict['bvp_prt_gauss'] = prt_gauss
        # Test PRT rr high frequency ponderation
        prt_rr_hf_pond = self.PRT_rr_hf_pond()
        features_dict['bvp_prt_rr_hf_pond'] = prt_rr_hf_pond

        # This place for PRT sd calculations
        # Test PRT sd1 feature
        prt_sd1_fsd1 = self.PRT_sd1_fsd1()
        features_dict['bvp_prt_sd1_fsd1'] = prt_sd1_fsd1
        # Test PRT sd1 T
        prt_sd1_T = self.PRT_sd1_T()
        features_dict['bvp_prt_sd1_T'] = prt_sd1_T
        # Test PRT sd2 feature
        prt_sd2_fsd2 = self.PRT_sd2_fsd2()
        features_dict['bvp_prt_sd2_fsd2'] = prt_sd2_fsd2
        # Test PRT sd2 L
        prt_sd2_L = self.PRT_sd2_L()
        features_dict['bvp_prt_sd2_L'] = prt_sd2_L
        # Test PRT sd2 csi feature
        prt_sd2_csi = self.PRT_sd2_csi()
        features_dict['bvp_prt_sd2_csi'] = prt_sd2_csi
        # Test PRT sd2 mcsi feature
        prt_sd2_mcsi = self.PRT_sd2_mcsi()
        features_dict['bvp_prt_sd2_mcsi'] = prt_sd2_mcsi
        # Test PRT sd2 cvi feature
        prt_sd2_cvi = self.PRT_sd2_cvi()
        features_dict['bvp_prt_sd2_cvi'] = prt_sd2_cvi
        # PRT sd calculations done

        # Test PA mean
        pa_mean = self.PA_mean()
        features_dict['bvp_pa_mean'] = pa_mean
        # Test PA low frequency quotient
        pa_lf_c = self.PA_lf_c()
        features_dict['bvp_pa_lf_c'] = pa_lf_c
        # Test PA hign frequency quotient
        pa_hf_c = self.PA_hf_c()
        features_dict['bvp_pa_hf_c'] = pa_hf_c
        # Test PA low frequency high frequency sum
        pa_lf_hf_s = self.PA_lf_hf_s()
        features_dict['bvp_pa_lf_hf_s'] = pa_lf_hf_s
        # Test PA low frequency high frequency quotient
        pa_lf_hf_c = self.PA_lf_hf_c()
        features_dict['bvp_pa_lf_hf_c'] = pa_lf_hf_c
        # Test PA gauss feature
        pa_gauss = self.PA_gauss()
        features_dict['bvp_pa_gauss'] = pa_gauss
        # Test PA rr high frequency ponderation
        pa_rr_hf_pond = self.PA_rr_hf_pond()
        features_dict['bvp_pa_rr_hf_pond'] = pa_rr_hf_pond
        # Test PWr mean
        pwr_mean = self.PWr_mean()
        features_dict['bvp_pwr_mean'] = pwr_mean
        # Test PWr low frequency quotient
        pwr_lf_c = self.PWr_lf_c()
        features_dict['bvp_pwr_lf_c'] = pwr_lf_c
        # Test PWr hign frequency quotient
        pwr_hf_c = self.PWr_hf_c()
        features_dict['bvp_pwr_hf_c'] = pwr_hf_c
        # Test PWr low frequency high frequency sum
        pwr_lf_hf_s = self.PWr_lf_hf_s()
        features_dict['bvp_pwr_lf_hf_s'] = pwr_lf_hf_s
        # Test PWr low frequency high frequency quotient
        pwr_lf_hf_c = self.PWr_lf_hf_c()
        features_dict['bvp_pwr_lf_hf_c'] = pwr_lf_hf_c
        # Test PWr gauss feature
        pwr_gauss = self.PWr_gauss()
        features_dict['bvp_pwr_gauss'] = pwr_gauss
        # Test PWr rr high frequency ponderation
        pwr_rr_hf_pond = self.PWr_rr_hf_pond()
        features_dict['bvp_pwr_rr_hf_pond'] = pwr_rr_hf_pond

        # This place for PWr sd calculations
        # Test PWr sd1 feature
        pwr_sd1_fsd1 = self.PWr_sd1_fsd1()
        features_dict['bvp_pwr_sd1_fsd1'] = pwr_sd1_fsd1
        # Test PWr sd1 T
        pwr_sd1_T = self.PWr_sd1_T()
        features_dict['bvp_pwr_sd1_T'] = pwr_sd1_T
        # Test PWr sd2 feature
        pwr_sd2_fsd2 = self.PWr_sd2_fsd2()
        features_dict['bvp_pwr_sd2_fsd2'] = pwr_sd2_fsd2
        # Test PWr sd2 L
        pwr_sd2_L = self.PWr_sd2_L()
        features_dict['bvp_pwr_sd2_L'] = pwr_sd2_L
        # Test PWr sd2 csi feature
        pwr_sd2_csi = self.PWr_sd2_csi()
        features_dict['bvp_pwr_sd2_csi'] = pwr_sd2_csi
        # Test PWr sd2 mcsi feature
        pwr_sd2_mcsi = self.PWr_sd2_mcsi()
        features_dict['bvp_pwr_sd2_mcsi'] = pwr_sd2_mcsi
        # Test PWr sd2 cvi feature
        pwr_sd2_cvi = self.PWr_sd2_cvi()
        features_dict['bvp_pwr_sd2_cvi'] = pwr_sd2_cvi
        # PWr sd calculations done

        return features_dict


if __name__ == '__main__':
    data_dict = mat73.loadmat('../BBDDLab_EH_CEI_VVG_IT07.mat')
    data_list = data_dict['BBDDLab_EH_CEI_VVG_IT07']
    bvp_data = data_list[0][0]['EH']['Video']['raw']['bvp_filt']
    sample_rate = 200
    # Parameters for feature
    feature_dict_all = {}
    len_frames = 3000
    overlap_frames = 2500
    ## Creating loop for extrcting the features maps
    start_index = 0
    for feature_index in range(int(len(bvp_data) / (len_frames - overlap_frames)) - 1):
        bvp_data_sample = bvp_data[start_index:start_index + len_frames]
        # Calculate SOP for BVP features
        Slope_list, Onset_list, Peak_list = SOP_del(bvp_data_sample, sample_rate)
        # define signal is raw_data_sample, frequency is 200
        bvp_features = BVP_features(signal=bvp_data_sample, slope_list=Slope_list, onset_list=Onset_list,
                                    peak_list=Peak_list, sampling_rate=sample_rate)
        # get features_dict
        features_dict = bvp_features.BVP_get_features_dic()
        feature_dict_all[str(feature_index)] = features_dict
        # overlap
        start_index = start_index + len_frames - overlap_frames
        #
        # for key, value in features_dict.items():
        #     print('{key}:{value}'.format(key=key, value=value))
        #
        # all_vlaue = {}
        # for k, v in features_dict.items():
        #     if v in all_vlaue:
        #         all_vlaue[v] = all_vlaue.get(v) + 1
        #     else:
        #         all_vlaue[v] = 1
        # print(all_vlaue)
        #
        print(len(features_dict.values()) == len(set(features_dict.values())))
    for key, value in feature_dict_all.items():
        print('{key}:{value}'.format(key=key, value=value))
        # print(len(value))
