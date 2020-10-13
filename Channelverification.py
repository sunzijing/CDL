
#import sys
#import re
#import os
#import math
import numpy as np
#from numpy import ceil
import scipy.fftpack as sp
import scipy.io as scio
#import datetime
#from sys import argv
import matplotlib.pyplot as plt

# 对信道模型进行PDP，Doppler,XPR,spatial correlation的验证
def	Channel_verification(delay, period, h_average, lOS_or_nLOS, channel_type,ma,na,am):
	temp = 'Verification finish'
	cluster_num = np.size(h_average, 2)
	cir_num = np.size(h_average, 3)
	delaypdp = (delay * 10 ** 2) / 10 ** 9
	calculation_point_num = 1099  # CTIA 1101个采样点，不足的补零
	sampling_rate = max(delaypdp) / calculation_point_num  # 根据最大时延计算采样率
	times = 1000
	window_coef = 1 / 2 * 7 / 8  # Kaiser窗系数
	cir_Interval = 10  # 统计间隔
	zero_Point = 2 # 1101，补零点
	channel_all_Cluster = np.squeeze(h_average[0, 0, :, :])
	n_t = zero_Point + round((max(delaypdp)) / sampling_rate)
	power = np.zeros((times, int(n_t)), dtype=float)
	for indexTimes in range(0, times):  # 对每个CIR采样点的多径信道采1101个点做fft在频域加窗处理，再回到时域，经过多个CIR点平均生成PDP
		time_domain_fn = np.zeros((cluster_num, int(n_t)), dtype=complex)
		selected_point = indexTimes * cir_Interval
		for clusterIndex in range(0, cluster_num):
			time_domain_index = int(round(delaypdp[clusterIndex] / sampling_rate) + round(zero_Point / 2))-1
			time_domain_fn[clusterIndex, time_domain_index] = channel_all_Cluster[clusterIndex, selected_point]
		time_domain_fn_sum = time_domain_fn.sum(axis=0)
		channel_fre = sp.fftshift(sp.fft(time_domain_fn_sum))  # 对每个样点的1101个采样点做fft
		fre_sweep = np.ones((1, int(n_t)), dtype=complex)
		index_l = int(round(n_t / 2 - n_t * window_coef))
		index_r = int(round(n_t / 2 + n_t * window_coef))
		window_len = index_r - index_l+1
		fre_sweep[:, index_l-1:index_r] = np.kaiser(window_len, 0.5)  # 频域加窗，集中簇中心的功率
		result_f = fre_sweep * channel_fre
		result_t = sp.ifft(sp.ifftshift(result_f))  # 加窗后回到时域
		power[indexTimes, :] = abs(result_t) ** 2  #计算功率，平均，归一化
	power = np.mean(power, axis=0)
	power = power/max(power)
	power = 10 * np.log10(power+np.spacing(1))
	#logbase = 10 * np.ones((1, int(n_t)), dtype=int)
	#power = 10 * np.log(power, logbase)
	t = np.linspace(0, n_t * sampling_rate, int(n_t))  # 调整时间采样点

	# 在时域计算信道链路上的PDP
	clusterpowertemp = np.zeros((1, cluster_num), dtype=float)
	h_average_temp = np.zeros((1, 1, cluster_num, cir_num), dtype=complex)
	h_average_temp[0, 0, :, :] = h_average[0, 0, :, :]
	if lOS_or_nLOS == 'LOS':
		h_average_temp[0, 0, 1, :] = h_average[0, 0, 1, :]+h_average[0, 0, 0, :]
		for cluster_index in range(1, cluster_num):
			clusterpowertemp[:, cluster_index] = np.mean(abs(np.squeeze(h_average_temp[0, 0, cluster_index, :]))**2)

		clusterpower=np.delete(clusterpowertemp, 0)
		delay_plot = delaypdp[1:]
	else:
		for cluster_index in range(0, cluster_num):
			clusterpowertemp[:, cluster_index] = np.mean(abs(np.squeeze(h_average_temp[0, 0, cluster_index, :]))**2)
		clusterpower = clusterpowertemp[0, :]
		delay_plot = delaypdp[0:]
	clusterpower = 10 * np.log10(clusterpower / (np.max(clusterpower) + np.spacing(1)))
    # 计算收发链路的多普勒谱

	channel_all_cluster = np.squeeze(h_average[0, 0, :, :])
	channel_sum_cluster = channel_all_cluster.sum(axis=0)
	window_len = 4/5
	window_num = np.ones((1, int(round(cir_num*window_len))), dtype=float)
	zero_point_num = np.zeros((1, cir_num-int(round(cir_num*window_len))), dtype=float)
	window = np.concatenate((window_num, zero_point_num), axis=1)
	tim_num = window*channel_sum_cluster
	fre_num = np.fft.fftshift(abs(np.fft.fft(tim_num)))  # 复数序列做fft后需进行fftshift进行调整
	fx = np.linspace(0, 1 / period, num=cir_num)
	#fx = np.arange(0, 1/period+1/period/(cir_num-1)-np.spacing(1), 1/period/(cir_num-1)) # 调整频域采样维度
	fx = (fx - (1/period)/2)[np.newaxis, :]
	fre_num = 10 * np.log10(fre_num+np.spacing(1))
	fx1 = fx[0, :]
	fre_num1 = fre_num[0, :]
	scio.savemat('Doppler.mat', {'fx': fx, 'fre_num': fre_num})
	'''
	#验证信道的交叉极化比
	xpr4cluster = np.zeros((1, cluster_num), dtype=float)
	if lOS_or_nLOS == 'LOS':
		for cluster_index in range(1, cluster_num):
			vv = np.sum(abs(np.squeeze(h_average[0, 0, cluster_index, :])+np.squeeze(h_average[0, int(ma*na/am), cluster_index, :]))**2)
			vh = np.sum(abs(np.squeeze(h_average[1, 0, cluster_index, :])+np.squeeze(h_average[1, int(ma*na/am), cluster_index, :]))**2)
			xpr4cluster[:, cluster_index] = 10 * np.log10(vv/(vh+np.spacing(1)))

	else:
		for cluster_index in range(0, cluster_num):
			vv = np.sum(abs(np.squeeze(h_average[0, 0, cluster_index, :])+np.squeeze(h_average[0, int(ma*na/am), cluster_index, :]))**2)
			vh = np.sum(abs(np.squeeze(h_average[1, 0, cluster_index, :])+np.squeeze(h_average[1, int(ma*na/am), cluster_index, :]))**2)
			xpr4cluster[:, cluster_index] = 10 * np.log10(vv/(vh+np.spacing(1)))
	print('XPR for each clusters')
	print(xpr4cluster)
	'''
	#验证信道的空间相关性
	pointnum = 1000  # CTIA中做相关时对应的样点数
	channel_fft_cv = np.zeros((pointnum, int(ma/am)), dtype=complex)  # 基站垂直12个阵子，对应12个位置点
	channel_fft_ch = np.zeros((pointnum, na), dtype=complex)   # 基站水平8个阵子，对应8个位置点
	channel_clusterv = np.zeros((cluster_num, pointnum, int(ma/am)), dtype=complex)
	channel_clusterh = np.zeros((cluster_num, pointnum, na), dtype=complex)
	channel_sum_cluster_v = np.zeros((int(ma/am), pointnum*int(round(np.max(delaypdp)/sampling_rate)+1)), dtype=complex)
	for ii_index in range(0, int(ma/am)):  #计算垂直维度以位置点和样点的2维位置样点矩阵
		channel_clusterv[:, :, ii_index] = h_average[0, ii_index, :, 0:cir_Interval*pointnum:cir_Interval]
		for index_num in range(0, pointnum):
			time_domain_cv = np.zeros((cluster_num, int(round(np.max(delaypdp)/sampling_rate))+1), dtype=complex)
			for clusterIndex in range(0, cluster_num):
				time_domain_index = int(round(delaypdp[clusterIndex]/sampling_rate))
				time_domain_cv[clusterIndex, time_domain_index] = channel_clusterv[clusterIndex, index_num, ii_index]
			channel_sum_cluster_v[ii_index, index_num*time_domain_cv.shape[1]:(index_num+1)*time_domain_cv.shape[1]] = time_domain_cv.sum(axis=0)
			fre_domain_cv_sum = sp.fftshift(sp.fft(time_domain_cv.sum(axis=0)))  #每个样点处按照簇错位相加后做fft
			channel_fft_cv[index_num, ii_index] = fre_domain_cv_sum[int(round((calculation_point_num+1)/2))-1] # 取中心频点处的值
	channel_sum_cluster_h = np.zeros((na, pointnum * int(round(np.max(delaypdp) / sampling_rate) + 1)), dtype=complex)
	for jj_index in range(0, na): #计算水平维度以位置点和样点的2维位置样点矩阵
		channel_clusterh[:, :, jj_index] = h_average[0, int(jj_index*ma/am), :, 0:cir_Interval*pointnum:cir_Interval]
		for index_num in range(0, pointnum):
			time_domain_ch = np.zeros((cluster_num, int(round(np.max(delaypdp)/sampling_rate))+1), dtype=complex)
			for clusterIndex in range(0, cluster_num):
				time_domain_index = int(round(delaypdp[clusterIndex]/sampling_rate))
				time_domain_ch[clusterIndex, time_domain_index] = channel_clusterh[clusterIndex, index_num, jj_index]
			channel_sum_cluster_h[jj_index, index_num  * time_domain_ch.shape[1]:(index_num+1) * time_domain_ch.shape[1]] = time_domain_ch.sum(axis=0)
			fre_domain_ch_sum = sp.fftshift(sp.fft(time_domain_ch.sum(axis=0)))  #每个样点处按照簇错位相加后做fft
			channel_fft_ch[index_num, jj_index] = fre_domain_ch_sum[int(round((calculation_point_num + 1) / 2))-1] # 取中心频点处的值

	corr_v_f = np.zeros((2, 2, int(ma/am)), dtype=float)
	corr_h_f = np.zeros((2, 2, na), dtype=float)
	corr_v_t = np.zeros((2, 2, int(ma/am)), dtype=float)
	corr_h_t = np.zeros((2, 2, na), dtype=float)
	# 不同位置上样点序列做相关
	for indv in range(0, int(ma/am)):
		corr_v_f[:, :, indv] = abs(np.corrcoef(channel_fft_cv[:, 0], channel_fft_cv[:, indv]))
		corr_v_t[:, :, indv] = abs(np.corrcoef(channel_sum_cluster_v[0, :], channel_sum_cluster_v[indv, :]))
	for indh in range(0, na):
		corr_h_f[:, :, indh] = abs(np.corrcoef(channel_fft_ch[:, 0], channel_fft_ch[:, indh]))
		corr_h_t[:, :, indh] = abs(np.corrcoef(channel_sum_cluster_h[0, :], channel_sum_cluster_h[indh, :]))
	correlation_v_f = np.squeeze(corr_v_f[0, 1, :])
	correlation_h_f = np.squeeze(corr_h_f[0, 1, :])
	correlation_v_t = np.squeeze(corr_v_t[0, 1, :])
	correlation_h_t = np.squeeze(corr_h_t[0, 1, :])
	if channel_type.lower() == 'CDL_A'.lower():
		#correlationV_theory = np.array([1.00000000000000,0.973974730183804,0.899862139822749,0.788756049808035,0.656669473400307,0.521396447406652,0.399135381835853,0.301603344839208,0.234245769299369,0.195886215838816,0.179825206547672,0.176061660482752])
		#correlationH_theory = np.array([1.00000000000000,0.964131630119955,0.863973674860465,0.719833607573338,0.559389435388085,0.410387809661578,0.293685429342913,0.218646578955052])
		correlationV_theory = np.array([1.00000000000000,0.302900301019998,0.567298565917483,0.579265119942349,0.316865070191705,0.324618857897414,0.269788807794478,0.175975383722289,0.102437821511352,0.166721261374764,0.0724272048031344,0.0996189004801771])
		correlationH_theory = np.array([1.00000000000000,0.430139746318321,0.585445909821662,0.392114354726269,0.347549761523416,0.333202043867174,0.304807924485806,0.123919373798378])

	elif channel_type.lower() == 'CDL_B'.lower():
		#correlationV_theory = np.array([1.00000000000000,0.969894411702889,0.884775375721201,0.758948480735496,0.612337914181607,0.465803549077708,0.336586695016880,0.235111480859317,0.163913152614971,0.118804479598289,0.0916818820031206,0.0738348182397860])
		#correlationH_theory = np.array([1.00000000000000,0.379056692553385,0.125532501601735,0.0671459952030588,0.142702289323440,0.0737724256507561,0.155069037608985,0.0560891815876677])
		correlationV_theory = np.array([1,0.911661636539493,0.701311473023904,0.500992508335124,0.407912389391648,0.365298163967221,0.296432636194522,0.205532656680734,0.123571727177351,0.0686546916643001,0.0426144878409035,0.0353691762848671])
		correlationH_theory = np.array([1,0.101536684649903,0.174180669022758,0.0415534993716735,0.100028082223433,0.00125577007366887,0.166882761183097,0.0153864603419366])

	elif channel_type.lower() == 'CDL_C'.lower():
		#correlationV_theory = np.array([1.00000000000000,0.969208259499543,0.882550839597750,0.755595128175814,0.609558226581870,0.465944045132416,0.341772562525911,0.246769220626128,0.183000752894312,0.146482413966663,0.129685274397138,0.123907241388023])
		#correlationH_theory = np.array([1.00000000000000,0.624019487731383,0.115816347599107,0.339298335989340,0.345254116593413,0.681155901988176,0.635538717081218,0.257416351290739])
		correlationV_theory = np.array([1.00000000000000,0.954006306706444,0.829067538021032,0.658515379932265,0.481930001427611,0.330635543486488,0.219834880193906,0.149566506748447,0.111337301572657,0.0951371114744061,0.0929587711109670,0.0983352718681467])
		correlationH_theory = np.array([1.00000000000000,0.430598515681197,0.261119970772915,0.133553418865956,0.381354205249463,0.479350878797527,0.541721231352851,0.201071282226093])


	elif channel_type.lower() == 'CDL_D'.lower():
		#correlationV_theory = np.array([1,0.996872805694556,0.988294633037767,0.976319918691782,0.963405019859734,0.951563535983685,0.941946475094896,0.934913578906064,0.930336714846675,0.927831765545025,0.926823718050266,0.926568608546551])
		#correlationH_theory = np.array([1,0.897751201811818,0.929601701950722,0.905107294131240,0.865896111336779,0.924714992137139,0.867108298065084,0.911712736058324])
		correlationV_theory = np.array([1.00000000000000,0.983922194084738,0.950377121160999,0.926983521486256,0.923673044969307,0.927435800728413,0.924599139780468,0.914448859230023,0.903694121530391,0.897315508114564,0.896332275914463,0.899349926686005])
		correlationH_theory = np.array([1.00000000000000,0.918661483737033,0.948390766538019,0.885749563182900,0.922737629269107,0.866488566573690,0.918642550841215,0.875882481419194])
	elif channel_type.lower() == 'CDL_E'.lower():
		correlationV_theory = np.array([1,0.994472546979698,0.979344978093545,0.958456909904667,0.936653036856074,0.918098918941537,0.905032944479814,0.897513240777885,0.894146462912289,0.893217930670682,0.893528737761277,0.894585178055987])
		correlationH_theory = np.array([0.997974589480394,0.901638630121989,0.911278532915258,0.883977489153742,0.862659118511598,0.917012417480444,0.873311903830371,0.912206658785614])
	else:
		correlationV_theory = np.array(
			[1.00000000000000, 0.969208259499543, 0.882550839597750, 0.755595128175814, 0.609558226581870,
			 0.465944045132416, 0.341772562525911, 0.246769220626128, 0.183000752894312, 0.146482413966663,
			 0.129685274397138, 0.123907241388023])
		correlationH_theory = np.array(
			[1.00000000000000, 0.624019487731383, 0.115816347599107, 0.339298335989340, 0.345254116593413,
			 0.681155901988176, 0.635538717081218, 0.257416351290739])

	# 绘制PDP，Doppler spectrum， Spatial correlation
	fig = plt.figure(figsize=(16, 8), facecolor='gray')
	fig1 = fig.add_subplot(2, 2, 1)
	fig1.plot(t*10**9, power, label='VNA')
	fig1.plot(delay_plot*10**9, clusterpower, 'ro', label='I/Q Analysis')
	fig1.set_title('PDP')
	fig1.set_xlabel('Time/ns')
	fig1.set_ylabel('Mag/dB')
	#plt.legend(loc='best', numpoints=1)
	plt.grid()
	fig2 = fig.add_subplot(2, 2, 2)
	fig2.plot(fx1, fre_num1, 'b', label='Doppler spectrum')
	fig2.set_title('Doppler spectrum')
	fig2.set_xlabel('fd Hz')
	fig2.set_ylabel('Mag/dB')
	fig2.set_ylim(1, np.max(fre_num)+20)
	fig2.set_xlim(-(1/period)/2, (1/period)/2)
	fig3 = fig.add_subplot(2, 2, 3)
	fig3.plot(np.arange(0, 12)*0.7, correlationV_theory, '*-r', label='Theory_V')
	fig3.plot(np.arange(0, int(ma/am))*0.7, correlation_v_t, 'b-+', label='I/Q analysis V')
	fig3.set_ylabel('Correlation_V')
	fig3.set_xlabel('Spatial distance[lambda]')
	plt.legend(loc='best')
	fig4 = fig.add_subplot(2, 2, 4)
	fig4.plot(np.arange(0, 8)*0.5, correlationH_theory, '*-r', label='Theory_H')
	fig4.plot(np.arange(0, na)*0.5, correlation_h_t, 'b-+', label='I/Q analysis H')
	fig4.set_ylabel('Correlation_H')
	fig4.set_xlabel('Spatial distance[lambda]')
	plt.legend(loc='best')

	plt.show()

	return temp



