import numpy as np
import mat73
from biosppy.signals import bvp
from scipy.signal import lombscargle
from distfit import distfit
import sys
import traceback


## Some functions need
def derivate_to_ternary(derivated_data_window):
    # Create an empty list to store the ternary data
    ternarized_data_window = []

    # Iterate through each element of the derivated_data_window
    for i in range(len(derivated_data_window)):

        # If derivative is positive, the function grows:
        if derivated_data_window[i] > 0:
            ternarized_data_window.append(1)

        # If derivative is negative, the function descends:
        elif derivated_data_window[i] < 0:
            ternarized_data_window.append(-1)

        # If derivative is null, the function is constant:
        elif derivated_data_window[i] == 0:
            ternarized_data_window.append(0)

    return ternarized_data_window


def find_local_maxima(ternarized_data_window):
    # Variable initialization
    Max_result = []

    # Iterate through each element of the ternarized_data_window
    for i in range(len(ternarized_data_window) - 1):

        # If it is a change from 1 to 0 or 1 to -1, then it is a local maxima.
        if ternarized_data_window[i] > ternarized_data_window[i + 1]:
            Max_result.append(i + 1)

    return Max_result


def find_local_minima(ternarized_data_window):
    # Variable initialization
    Min_result = []

    # Iterate through each element of the ternarized_data_window
    for i in range(len(ternarized_data_window) - 1):

        # If it is a change from 1 to 0 or 1 to -1, then it is a local maxima.
        if ternarized_data_window[i] < ternarized_data_window[i + 1]:
            Min_result.append(i + 1)

    return Min_result


## SOP_del FUNCTION
# This function obtains the BVP main biomarkers: the Onset, the Slope and
# the Peak biomarker, which are used for further delineations. If any
# feature is needed to be extracted from this signal, this delineation needs
# always to be done.
# How this function works is by using a sliding window that calculates a
# series of local maxima and minima, derivating the signal, and thanks to a
# series of thresholds, obtains the position of the biomarkers.

# Inputs for this function:
# -data_window: the acquisition window of the raw (or filtered) BVP signal
# from which the biomarkers are going to be extracted.
# -Fs: Sample frequency of the analysed signal.
# -start: index used to calculate the actual position of the biomarker,
# acordingly to the complete signal.

# Outputs for this function:
# -Slope: Vector where the extracted Slope biomarkers are saved.
# -Peak: Vector where the extracted Peak biomarkers are saved.
# -Onset: Vector where the extracted Onset biomarkers are saved.

## Have not known how to extract data_window
def SOP_del(data_window, Fs, start=0):
    window_length_secs = 1.3  # In seconds, is the size of the sliding window.
    nof_overlap_secs = 1  # In seconds, is the size of the overlapping for the sliding window.
    th_amp_bvp = 0.9  # Threshold, it is a multiplier for the biomarker calculation.
    min_fp_bvp = 0.33  # Threshold, it is a minimum distance between two consecutive biomarkers
    max_fp_bvp = 1.7  # Threshold, it is a maximum distance between two consecutive biomarkers
    max_onset_peak_bvp = 0.8  # Threshold, it is a maximum distance between two consecutive Peak and Onset biomarkers.
    min_onset_peak_bvp = 0.06  # Threshold, it is a minimum distance between two consecutive Peak and Onset biomarkers.
    local_min_neg_th = 0.3  # Threshold, it is the local minima minimum value.
    local_min_pos_th = 1.7  # Threshold, it is the local minima maximum value.
    max_consecutive_peak_bvp = 0.05  # Special flag if a condition is reached, so a series of biomarkers found behave
    # as they should.

    # Where the found biomarkers are saved:
    Slope = []
    Onset = []
    Peak = []
    # Variable where the size (in samples) of the acquisition window is saved.
    window_size = len(data_window)
    ## Usefull data precalculation:
    first_derivate_data_window = np.diff(data_window)
    second_derivate_data_window = np.diff(first_derivate_data_window)
    ternarized_data_window = derivate_to_ternary(second_derivate_data_window)
    Max_result = find_local_maxima(ternarized_data_window)
    Max_result_array = np.array(Max_result)
    increment = int((window_length_secs - nof_overlap_secs) * Fs)
    N = window_size - 1
    nof_segments = int((N - window_length_secs * Fs) / increment)
    window_length = int(window_length_secs * Fs)

    w_start = 0
    w_end = window_length
    l = 0

    ## Slope Calculation
    if N > window_length:
        # We go through every segment in which the acquisition window has
        # been divided.
        for q in range(nof_segments):
            # Overflow comprobation
            # if (w_start + w_end - 2) > N:
            #     # Overflow start and end indexes actualization
            #     w_start = N - window_length
            #     w_end = N - 1

            # mean for the shifting window raw signal:
            window_mean = np.mean(first_derivate_data_window[w_start:w_end])

            # std for the shifting window raw signal:
            window_std = np.std(first_derivate_data_window[w_start:w_end])

            # Main Slope threshold calculation:
            bvp_window_threshold = (window_mean + window_std) * th_amp_bvp

            # Flag used if a maximum value is found:
            flag_max = 0

            # We go through every Max found in the shifting window being
            # analysed:
            for j in range(len(Max_result)):
                # Check the max values found in the shifting window only:
                if (Max_result[j] >= w_start) and (Max_result[j] <= w_end):
                    # Check the max value with the threshold previously
                    # calculated:
                    if first_derivate_data_window[Max_result[j]] >= bvp_window_threshold:
                        # If it is the first Slope found:
                        if l == 0:
                            # Slope found
                            Slope.append(Max_result[j])

                            # We have a Slope biomarker.
                            l += 1

                            # Reset flag
                            flag_max = 0
                        else:
                            # We go through every previously found Slope
                            # biomarkers and check that value with the
                            # currently being analysed value:
                            for k in range(l - 1):
                                if Max_result[j] == Slope[k]:
                                    flag_max += 1

                            # If the found value did not match with any of
                            # the other Slopes found, then we may have found
                            # a new Slope:
                            if flag_max == 0:
                                # Threshold check for minimum:
                                if (Max_result[j] - Slope[l - 1]) > (min_fp_bvp * Fs):
                                    # Threshold check for maximum:
                                    if (Max_result[j] - Slope[l - 1]) > (max_fp_bvp * Fs):
                                        # Out of bounds. The found biomarker
                                        # is not a Slope. The next Max value,
                                        # besides, is.
                                        Slope.append(0)  # Null value resembles a non valid found Slope.
                                        l += 1
                                        Slope.append(Max_result[j])
                                        l += 1
                                    else:
                                        # In bounds. The found biomarker is
                                        # a Slope.
                                        Slope.append(Max_result[j])
                                        l += 1

                            # Reset flag
                            flag_max = 0

            # Indexes actualization:
            w_start += increment
            w_end += increment

    ## Peak and Onset calculation:
    min_val = 0
    max_val = 0

    for a in range(len(Slope)):

        # Flag initialization:
        max_detect = 0

        # If the current Slope being analysed is not valid, so they are the Onset and Slope.
        if Slope[a] == 0:
            Onset.append(0)
            Peak.append(0)

        # If not, we search
        else:

            # First Slope found being analysed:
            if a == 0:

                # Various threshold and flag assignations:
                pos_before_onset = Slope[a] - int((max_onset_peak_bvp / 1.9) * Fs)
                prev_signal_der = 2

            # If it is not the first found Slope, and it is a valid one:
            elif Slope[a - 1] != 0:

                # Threshold assignation:
                pos_before_onset = Slope[a] - int((Slope[a] - Slope[a - 1]) / 1.9)

            # If it is not the first found Slope, and it is NOt a valid one:
            elif Slope[a - 1] == 0:

                # Various threshold and flag assignations:
                pos_before_onset = Slope[a] - int((max_onset_peak_bvp / 1.9) * Fs)
                prev_signal_der = 2

            # Not the first Slope found being analysed:
            if a > 0:

                # Threshold comprobation:
                if pos_before_onset < Slope[a - 1]:
                    # Threshold reassignation:
                    pos_before_onset = Slope[a - 1] + 1

            # Threshold value comprobation
            if pos_before_onset <= 0:
                # Threshold reassignation:
                pos_before_onset = 1

            # If it is the last Slope found the one being analysed:
            if a == len(Slope) - 1:

                # Threshold assignation:
                locsprev = window_size

            # If not:
            else:

                # Threshold assignation:
                locsprev = Slope[a] + int(max_onset_peak_bvp * Fs)

                # Threshold value comprobation
                if locsprev > window_size:

                    # Threshold reassigntaion
                    locsprev = window_size

                elif locsprev > Slope[a + 1]:
                    if Slope[a + 1] != 0:
                        # Threshold reassigntaion
                        locsprev = Slope[a + 1] - 1

            for b in range(pos_before_onset + 1, locsprev):
                # Calculate slope for certain point:
                signal_der = data_window[b] - data_window[b - 1]

                # There is an increment:
                if signal_der > 0:
                    signal_der = 1
                # The signal is constant:
                elif signal_der == 0:
                    if prev_signal_der == 1:
                        signal_der = 1  # Same value as before
                    else:
                        signal_der = 2  # Same value as before
                # There is a decrement:
                else:
                    signal_der = 2

                # Analysis before de Slope (we are looking for the Onset biomarker):
                if b < Slope[a]:
                    # If it is the first analyzed value, I save it as a minimum
                    if b == (pos_before_onset + 1):
                        min_val = b
                    # If not, we check
                    else:
                        # Is there a local minimum?
                        if prev_signal_der == 2 and signal_der == 1:
                            # If previous minimum is negative
                            if data_window[min_val] < 0:
                                # Threshold comprobation
                                if data_window[b] < (data_window[min_val] * local_min_neg_th):
                                    # We found an Onset:
                                    min_val = b
                            else:
                                # Threshold comprobation
                                if data_window[b] < (data_window[min_val] * local_min_pos_th):
                                    # We found an Onset:
                                    min_val = b
                else:
                    if a == len(Slope):
                        max_val = 0
                    else:
                        if b == Slope[a]:
                            max_val = Slope[a]
                        else:
                            if (prev_signal_der == 1) and (signal_der == 2):
                                # If we have not found a local maxima yet:
                                if max_detect == 0:
                                    max_val = b
                                    max_detect = max_val + (max_consecutive_peak_bvp * Fs)
                                # If we have, we check its value:
                                elif b <= max_detect:
                                    # If there is a higher value, then that is the Peak (local maxima):
                                    if data_window[b] > data_window[max_val]:
                                        max_val = b
                prev_signal_der = signal_der

            Peak.append(max_val)
            Onset.append(min_val)

            # Special consideration check
            if a < len(Slope):
                # Threshold comprobation: If it is true, the found biomarkers are considered not valid
                if ((max_onset_peak_bvp * Fs) < (Peak[a - 1] - Onset[a - 1])) or (
                        (Peak[a - 1] - Onset[a - 1]) < (min_onset_peak_bvp * Fs)):
                    Slope[a - 1] = 0
                    Peak[a - 1] = 0
                    Onset[a - 1] = 0
    # Delete last Peak and Slope (We need one more Onset than Slope and Peak):
    if len(Onset) == len(Slope):
        if Onset:
            Slope.pop()
            Peak.pop()

    # Non valid found biomarkers deletion:
    find_Onset = [i for i, val in enumerate(Onset) if val == 0]
    Onset = [val for i, val in enumerate(Onset) if i not in find_Onset]
    find_Peak = [i for i, val in enumerate(Peak) if val == 0]
    Peak = [val for i, val in enumerate(Peak) if i not in find_Peak]
    find_Slope = [i for i, val in enumerate(Slope) if val == 0]
    Slope = [val for i, val in enumerate(Slope) if i not in find_Slope]

    # Real value assignation using its actual position:
    Onset = [val + start for val in Onset]
    Peak = [val + start for val in Peak]
    Slope = [val + start for val in Slope]

    return Slope, Onset, Peak


if __name__ == '__main__':
    data_dict = mat73.loadmat('../BBDDLab_EH_CEI_VVG_IT07.mat')
    data_list = data_dict['BBDDLab_EH_CEI_VVG_IT07']
    raw_data_sample = data_list[0][0]['EH']['Video']['raw']

    # Compute the derivative of the filtered signal
    bvp_derivative = raw_data_sample['bvp'][:3000]

    Slope, Onset, Peak = SOP_del(bvp_derivative, 200)
    print(1)
