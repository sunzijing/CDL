
# # 	******************************************************************************/

#import sys
#import re
#import os
#import math
import numpy as np
import pandas as pands
#import scipy
#import pickle
#import datetime
#import time
import scipy.io as scio
#from sys import argv
import matplotlib.pyplot as plt

################################################################################### 基础参配置 ###########################################################
########################################################################################################################################################
Basic_config_parameters = pands.read_csv('./Basic_config_parameters.csv')   #暂时从根目录的excel中导入模型基本参数
CentralFrequence= Basic_config_parameters['CentralFrequence'].loc[0]             #用于画图，存储
LightSpeed = 2.99792458*10**8 				# speed of light
WaveLength = LightSpeed / CentralFrequence  ##存储
k_CONST = 2*np.pi/WaveLength  ##存储
CentralFrequence = Basic_config_parameters['CentralFrequence'].loc[0]  ##存储
BS_Num = Basic_config_parameters['BS_Num'].loc[0]    ##存储
BS_Num = Basic_config_parameters['UE_Num'].loc[0]  ##存储
Emulator_type = Basic_config_parameters['Emulator_type'].loc[0].lower()  #导入信道模拟器类型，存储
Running_time = Basic_config_parameters['Running_time'].loc[0]  ##存储
V_max = Basic_config_parameters['V_max_km'].loc[0]  ##存储
V_max = V_max/3.6
if Emulator_type == 'f64':
    SD = 4     # F64采样密度参数，详情见仪表使用说明
    CIR_update_Rate = 2 * SD * V_max * CentralFrequence / LightSpeed  # F64 CIR 采样率
    period = 1 / CIR_update_Rate   ##存储
    P_num = np.round(Running_time/period)   ##存储
elif Emulator_type == 'ksw':
    SD = 8
    Ts = 2 * SD * V_max * CentralFrequence / LightSpeed  #KSW 默认采样率，其中采样目的暂时固定位16
    period = 1/Ts  ##存储
    P_num = 32 * np.round(Running_time/period /32)  ##存储						# 导入系数n，CIR样点数必须为32的整数倍 TAG！
else:    #错误仪表标识采用默认的ksw配置
    SD = 8
    Ts = 2 * SD * V_max * CentralFrequence / LightSpeed  #KSW 默认采样率，其中采样目的暂时固定位16
    period = 1/Ts    ##存储
    P_num = 32 * np.round(Running_time/period /32)  ##存储						# 导入系数n，CIR样点数必须为32的整数倍 TAG！
################################################################################### 基站与终端参数配置(最大支持12个BS和16UE) ###########################################################
########################################################################################################################################################
#####################BS######################################
Configurations_for_BS1 = pands.read_csv('./Configurations_for_BS1.csv')   #暂时从根目录的excel中导入模型基本参数
CoordinateBS1_X = Configurations_for_BS1['X_m'].loc[0]             #用于画图，存储
CoordinateBS1_Y = Configurations_for_BS1['Y_m'].loc[0]             #用于画图，存储
CoordinateBS1_Z = Configurations_for_BS1['Z_m'].loc[0]             #用于画图，存储
alpha_BS = Configurations_for_BS1['alpha_BS'].loc[0]             #用于真实阵子坐标计算，存储
Beta_BS= Configurations_for_BS1['Beta_BS'].loc[0]             #用于真实阵子坐标计算，存储
Gamma_BS = Configurations_for_BS1['Gamma_BS'].loc[0]             #用于真实阵子坐标计算，存储
##########################################阵子排布与下倾################################
S = Configurations_for_BS1['S'].loc[0]  									# 存储
M_BS = Configurations_for_BS1['M_BS'].loc[0] 										# BS垂直同向振子个数；存储
N_BS = Configurations_for_BS1['N_BS'].loc[0]										# BS水平同向振子个数；存储
Md_BS = Configurations_for_BS1['BS_D_v_lambda'].loc[0]										# BS水平同向振子个数；存储
Nd_BS = Configurations_for_BS1['BS_D_h_lambda'].loc[0]
AE_Comb = Configurations_for_BS1['AE_Comb'].loc[0]										#振子->通道合路数；存储
Theta_downtilt = Configurations_for_BS1['Theta_downtilt'].loc[0]										#下倾角；存储
###### ###############3GPP 38.901振子天线方向图参数,暂时固定，用于画图####################

SLA_v = 30									#dB，存储
A_max = 30									#dB，存储
theta_3dB = 65								#degree，存储
phi_3dB = 65									#degree，存储
theta_Beam = 90								#degree，存储





########################################基站和的极化斜角赋值###########################
BSPolarizationPlacement = Configurations_for_BS1['BSPolarizationPlacement'].loc[0] 			# Cross ；Single;存储
		
BS_slant1 = Configurations_for_BS1['BS_slant1'].loc[0] 			# Cross ；Single;存储
BS_slant2= Configurations_for_BS1['BS_slant2'].loc[0] 			# Cross ；Single;存储

###################计算基站天线阵子坐标,用于画图##################
###########Mp->M_BS;Np->N_BS;Md->Md_BS;Nd->Nd_BS
'''
BsElementPosition = np.zeros((int(S), 3), dtype=float)    # 基站侧阵子坐标

if BSPolarizationPlacement.lower() == 'BSCross'.lower() :  #双极化阵子排布
	location_antennatemp = np.array(np.zeros((1, int(S / 2 * 3))), dtype=float)
	antenna_index = 0
	for numberantenna_H in range(0, Np, 1):
		for numberantenna_V in range(0, Mp, 1):
			location_antennatemp[0, antenna_index * 3] = 0
			location_antennatemp[0, antenna_index * 3 + 1] = numberantenna_H * Nd
			location_antennatemp[0, antenna_index * 3 + 2] = numberantenna_V * Md
			antenna_index = antenna_index + 1
	location_antennatemp = location_antennatemp.reshape(int(S / 2), 3)
	BsElementPosition[0:int(S / 2):1, :] = location_antennatemp
	BsElementPosition[int(S / 2):S:1, :] = location_antennatemp
	location_antennatemp = []
else:   #单极化情况阵子排布
	location_antennatemp = np.array(np.zeros((1, int(S * 3))), dtype=float)
	antenna_index = 0
	for numberantenna_H in range(0, Np, 1):
		for numberantenna_V in range(0, Mp, 1):
			location_antennatemp[0, antenna_index * 3] = 0
			location_antennatemp[0, antenna_index * 3 + 1] = numberantenna_H * Nd
			location_antennatemp[0, antenna_index * 3 + 2] = numberantenna_V * Md
			antenna_index = antenna_index + 1
	location_antennatemp = location_antennatemp.reshape(S, 3)
	BsElementPosition = location_antennatemp
	location_antennatemp = []
'''
#######################################终端参数###############################################
Configurations_for_UE1 = pands.read_csv('./Configurations_for_UE1.csv')   #暂时从根目录的excel中导入模型基本参数
CoordinateUE1_X = Configurations_for_UE1['X_m'].loc[0]             #用于画图，存储
CoordinateUE1_Y = Configurations_for_UE1['Y_m'].loc[0]             #用于画图，存储
CoordinateUE1_Z = Configurations_for_UE1['Z_m'].loc[0]             #用于画图，存储
alpha_UE = Configurations_for_UE1['alpha_UE'].loc[0]             #用于真实阵子坐标计算，存储
Beta_UE= Configurations_for_UE1['Beta_UE'].loc[0]             #用于真实阵子坐标计算，存储
Gamma_UE = Configurations_for_UE1['Gamma_UE'].loc[0]             #用于真实阵子坐标计算，存储
U = Configurations_for_UE1['U'].loc[0]  # CE output antenna element number
M_UE = Configurations_for_UE1['M_UE'].loc[0]   # #单极化垂直阵子数；存储
N_UE = Configurations_for_UE1['N_UE'].loc[0]   # #单极化水平阵子数；存储
Md_UE = Configurations_for_UE1['UE_D_v_lambda'].loc[0] * WaveLength  # #UE垂直振子间距；存储
Nd_UE = Configurations_for_UE1['UE_D_h_lambda'].loc[0] * WaveLength  # #UE水平振子间距；存储
V_km = Configurations_for_UE1['V_km'].loc[0]
V = V_km / 3.6  # 存储
Trace_UE = Configurations_for_UE1['Trace'].loc[0]  #存储,直线，弧线
Acc = Configurations_for_UE1['Acc_km_s'].loc[0] /3.6 #加速度m/s^2，存储
	########################################基站和终端的极化斜角赋值###########################
UEPolarizationPlacement = Configurations_for_UE1['UEPolarizationPlacement'].loc[0]   # Cross ；Single;存储
UE_slant1 = Configurations_for_UE1['UE_slant1'].loc[0] 			# Cross ；Single;存储
UE_slant2= Configurations_for_UE1['UE_slant2'].loc[0] 			# Cross ；Single;存储

'''
###########Mpu->M_UE;Npu->N_UE;Md->Md_UE;Nd->Nd_UE
	#####################计算终端天线阵子坐标################

	MsElementPosition = np.zeros((int(U), 3), dtype=float)  # 终端阵子坐标
	if UEPolarizationPlacement.lower() == 'UECross'.lower():
		location_antennatemp = np.array(np.zeros((1, int(U / 2 * 3))), dtype=float)
		antenna_index = 0
		for numberantenna_H in range(0, Npu, 1):
			for numberantenna_V in range(0, Mpu, 1):
				location_antennatemp[0, antenna_index * 3] = 0
				location_antennatemp[0, antenna_index * 3 + 1] = numberantenna_H * Ndu
				location_antennatemp[0, antenna_index * 3 + 2] = numberantenna_V * Mdu
				antenna_index = antenna_index + 1
		location_antennatemp = location_antennatemp.reshape(int(U / 2), 3)
		MsElementPosition[0:U:2, :] = location_antennatemp
		MsElementPosition[1:U + 1:2, :] = location_antennatemp
		location_antennatemp = []

	else:
		location_antennatemp = np.array(np.zeros((1, int(U * 3))), dtype=float)
		antenna_index = 0
		for numberantenna_H in range(0, Npu, 1):
			for numberantenna_V in range(0, Mpu, 1):
				location_antennatemp[0, antenna_index * 3] = 0
				location_antennatemp[0, antenna_index * 3 + 1] = numberantenna_H * Ndu
				location_antennatemp[0, antenna_index * 3 + 2] = numberantenna_V * Mdu
				antenna_index = antenna_index + 1
		location_antennatemp = location_antennatemp.reshape(int(U), 3)
		MsElementPosition = location_antennatemp
		location_antennatemp = []
'''

###################################################################################信道模型参数配置 ###########################################################
############################################################################################################################################################
#####################38.901 CDL 参数集################

Channelmodel_parameters = pands.read_csv('./Channelmodel_parameters.csv')   #暂时从根目录的excel中导入模型基本参数

Modeling_scheme = Channelmodel_parameters['Modeling_scheme'].loc[0]
Largescale_type= Channelmodel_parameters['Largescale_type'].loc[0]
smallscale_type = Channelmodel_parameters['smallscale_type'].loc[0]
Raycoupling_type = Channelmodel_parameters['Raycoupling_type'].loc[0]

######################################CDL Table初始化存储#############################
if smallscale_type.lower() == 'CDL_A'.lower() or smallscale_type.lower() == 'CDL_B'.lower() or smallscale_type.lower() == 'CDL_C'.lower():  # case  insensitive
	LOS_or_NLOS = 'NLOS'
else:
	LOS_or_NLOS = 'LOS'

LOS = 'LOS'
rayNum = 20
#####################################下列操作可封装为类的成员函数，函数功能为可通过CDL类型存储下列变量成员，返回table类型并存储#################
if smallscale_type.lower() == 'CDL_A'.lower():
	power = np.array(
		[-13.4, 0, -2.2, -4, -6, -8.2, -9.9, -10.5, -7.5, -15.9, -6.6, -16.7, -12.4, -15.2, -10.8, -11.3, -12.7, -16.2,
		 -18.3, -18.9, -16.6, -19.9, -29.7])
	delay = np.array(
		[0.0, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708, 0.5750, 0.7618, 1.5375, 1.8978, 2.2242, 2.1718, 2.4942,
		 2.5119, 3.0582, 4.0810, 4.4579, 4.5695, 4.7966, 5.0066, 5.3043, 9.6586])
	AOD = np.array(
		[-178.1, -4.2, -4.2, -4.2, 90.2, 90.2, 90.2, 121.5, -81.7, 158.4, -83, 134.8, -153, -172, -129.9, -136, 165.4,
		 148.4, 132.7, -118.6, -154.1, 126.5, -56.2])
	AOA = np.array(
		[51.3, -152.7, -152.7, -152.7, 76.6, 76.6, 76.6, -1.8, -41.9, 94.2, 51.9, -115.9, 26.6, 76.6, -7, -23, -47.2,
		 110.4, 144.5, 155.3, 102, -151.8, 55.2])
	ZOD = np.array(
		[50.2, 93.2, 93.2, 93.2, 122, 122, 122, 150.2, 55.2, 26.4, 126.4, 171.6, 151.4, 157.2, 47.2, 40.4, 43.3, 161.8,
		 10.8, 16.7, 171.7, 22.7, 144.9])
	ZOA = np.array(
		[125.4, 91.3, 91.3, 91.3, 94, 94, 94, 47.1, 56, 30.1, 58.8, 26, 49.2, 143.1, 117.4, 122.7, 123.2, 32.6, 27.2,
		 15.2, 146, 150.7, 156.1])
	ASD = 5
	ASA = 11
	ZSD = 3
	ZSA = 3
	XPR = 10
	Delay_spread = 100

elif smallscale_type.lower() == 'CDL_B'.lower():
	power = np.array(
		[0, -2.2, -4, -3.2, -9.8, -1.2, -3.4, -5.2, -7.6, -3, -8.9, -9, -4.8, -5.7, -7.5, -1.9, -7.6, -12.2, -9.8,
		 -11.4, -14.9, -9.2, -11.3])
	delay = np.array(
		[0, 0.1072, 0.2155, 0.2095, 0.2870, 0.2986, 0.3752, 0.5055, 0.3681, 0.3697, 0.5700, 0.5283, 1.1021, 1.2756,
		 1.5474, 1.7842, 2.0169, 2.8294, 3.0219, 3.6187, 4.1067, 4.2790, 4.7834])
	AOD = np.array(
		[9.3, 9.3, 9.3, -34.1, -65.4, -11.4, -11.4, -11.4, -67.2, 52.5, -72, 74.3, -52.2, -50.5, 61.4, 30.6, -72.5,
		 -90.6, -77.6, -82.6, -103.6, 75.6, -77.6])
	AOA = np.array(
		[-173.3, -173.3, -173.3, 125.5, -88, 155.1, 155.1, 155.1, -89.8, 132.1, -83.6, 95.3, 103.7, -87.8, -92.5,
		 -139.1, -90.6, 58.6, -79.0, 65.8, 52.7, 88.7, -60.4])
	ZOD = np.array(
		[105.8, 105.8, 105.8, 115.3, 119.3, 103.2, 103.2, 103.2, 118.2, 102.0, 100.4, 98.3, 103.4, 102.5, 101.4, 103,
		 100, 115.2, 100.5, 119.6, 118.7, 117.8, 115.7])
	ZOA = np.array(
		[78.9, 78.9, 78.9, 63.3, 59.9, 67.5, 67.5, 67.5, 82.6, 66.3, 61.6, 58.0, 78.2, 82.0, 62.4, 78.0, 60.9, 82.9,
		 60.8, 57.3, 59.9, 60.1, 62.3])
	ASD = 10
	ASA = 22
	ZSD = 3
	ZSA = 7
	XPR = 8
	Delay_spread = 100

elif smallscale_type.lower() == 'CDL_C'.lower():
	power = np.array(
		[-4.4, -1.2, -3.5, -5.2, -2.5, 0, -2.2, -3.9, -7.4, -7.1, -10.7, -11.1, -5.1, -6.8, -8.7, -13.2, -13.9, -13.9,
		 -15.8, -17.1, -16, -15.7, -21.6, -22.8])
	delay = np.array(
		[0, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.6560, 0.6584, 0.7935, 0.8213, 0.9336, 1.2285, 1.3083,
		 2.1704, 2.7105, 4.2589, 4.6003, 5.4902, 5.6077, 6.3065, 6.6374, 7.0427, 8.6523])
	AOD = np.array(
		[-46.6, -22.8, -22.8, -22.8, -40.7, 0.3, 0.3, 0.3, 73.1, -64.5, 80.2, -97.1, -55.3, -64.3, -78.5, 102.7, 99.2,
		 88.8, -101.9, 92.2, 93.3, 106.6, 119.5, -123.8])
	AOA = np.array(
		[-101, 120, 120, 120, -127.5, 170.4, 170.4, 170.4, 55.4, 66.5, -48.1, 46.9, 68.1, -68.7, 81.5, 30.7, -16.4, 3.8,
		 -13.7, 9.7, 5.6, 0.7, -21.9, 33.6])
	ZOD = np.array(
		[97.2, 98.6, 98.6, 98.6, 100.6, 99.2, 99.2, 99.2, 105.2, 95.3, 106.1, 93.5, 103.7, 104.2, 93.0, 104.2, 94.9,
		 93.1, 92.2, 106.7, 93.0, 92.9, 105.2, 107.8])
	ZOA = np.array(
		[87.6, 72.1, 72.1, 72.1, 70.1, 75.3, 75.3, 75.3, 67.4, 63.8, 71.4, 60.5, 90.6, 60.1, 61.0, 100.7, 62.3, 66.7,
		 52.9, 61.8, 51.9, 61.7, 58, 57])
	ASD = 2
	ASA = 15
	ZSD = 3
	ZSA = 7
	XPR = 7
	Delay_spread = 100

elif smallscale_type.lower() == 'CDL_D'.lower():

	power = np.array([-0.2, -13.5, -18.8, -21, -22.8, -17.9, -20.1, -21.9, -22.9, -27.8, -23.6, -24.8, -30.0, -27.7])
	delay = np.array([0, 0, 0.035, 0.612, 1.363, 1.405, 1.804, 2.596, 1.775, 4.042, 7.937, 9.424, 9.708, 12.525])
	AOD = np.array([0, 0, 89.2, 89.2, 89.2, 13, 13, 13, 34.6, -64.5, -32.9, 52.6, -132.1, 77.2])
	AOA = np.array([-180, -180, 89.2, 89.2, 89.2, 163, 163, 163, -137, 74.5, 127.7, -119.6, -9.1, -83.8])
	ZOD = np.array([98.5, 98.5, 85.5, 85.5, 85.5, 97.5, 97.5, 97.5, 98.5, 88.4, 91.3, 103.8, 80.3, 86.5])
	ZOA = np.array([81.5, 81.5, 86.9, 86.9, 86.9, 79.4, 79.4, 79.4, 78.2, 73.6, 78.3, 87, 70.6, 72.9])
	ASD = 5
	ASA = 8
	ZSD = 3
	ZSA = 3
	XPR = 11
	Delay_spread = 100

elif smallscale_type.lower() == 'CDL_E'.lower():

	power = np.array(
		[-0.03, -22.03, -15.8, -18.1, -19.8, -22.9, -22.4, -18.6, -20.8, -22.6, -22.3, -25.6, -20.2, -29.8, -29.2])
	delay = np.array(
		[0, 0, 0.5133, 0.5440, 0.5630, 0.5440, 0.7112, 1.9092, 1.9293, 1.9589, 2.6426, 3.7136, 5.4524, 12.0034,
		 20.6419])
	AOD = np.array([0, 0, 57.5, 57.5, 57.5, -20.1, 16.2, 9.3, 9.3, 9.3, 19, 32.7, 0.5, 55.9, 57.6])
	AOA = np.array([-180, -180, 18.2, 18.2, 18.2, 101.8, 112.9, -155.5, -155.5, -155.5, -143.3, -94.7, 147, -36.2, -26])
	ZOD = np.array([99.6, 99.6, 104.2, 104.2, 104.2, 99.4, 100.8, 98.8, 98.8, 98.8, 100.8, 96.4, 98.9, 95.6, 104.6])
	ZOA = np.array([80.4, 80.4, 80.4, 80.4, 80.4, 80.8, 86.3, 82.7, 82.7, 82.7, 82.9, 88, 81, 88.6, 78.3])
	ASD = 5
	ASA = 11
	ZSD = 3
	ZSA = 7
	XPR = 8
	Delay_spread = 100

else:
	power = np.array(
		[-4.4, -1.2, -3.5, -5.2, -2.5, 0, -2.2, -3.9, -7.4, -7.1, -10.7, -11.1, -5.1, -6.8, -8.7, -13.2, -13.9, -13.9,
		 -15.8, -17.1, -16, -15.7, -21.6, -22.8])
	delay = np.array(
		[0, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.6560, 0.6584, 0.7935, 0.8213, 0.9336, 1.2285, 1.3083,
		 2.1704, 2.7105, 4.2589, 4.6003, 5.4902, 5.6077, 6.3065, 6.6374, 7.0427, 8.6523])
	AOD = np.array(
		[-46.6, -22.8, -22.8, -22.8, -40.7, 0.3, 0.3, 0.3, 73.1, -64.5, 80.2, -97.1, -55.3, -64.3, -78.5, 102.7, 99.2,
		 88.8, -101.9, 92.2, 93.3, 106.6, 119.5, -123.8])
	AOA = np.array(
		[-101, 120, 120, 120, -127.5, 170.4, 170.4, 170.4, 55.4, 66.5, -48.1, 46.9, 68.1, -68.7, 81.5, 30.7, -16.4, 3.8,
		 -13.7, 9.7, 5.6, 0.7, -21.9, 33.6])
	ZOD = np.array(
		[97.2, 98.6, 98.6, 98.6, 100.6, 99.2, 99.2, 99.2, 105.2, 95.3, 106.1, 93.5, 103.7, 104.2, 93.0, 104.2, 94.9,
		 93.1, 92.2, 106.7, 93.0, 92.9, 105.2, 107.8])
	ZOA = np.array(
		[87.6, 72.1, 72.1, 72.1, 70.1, 75.3, 75.3, 75.3, 67.4, 63.8, 71.4, 60.5, 90.6, 60.1, 61.0, 100.7, 62.3, 66.7,
		 52.9, 61.8, 51.9, 61.7, 58, 57])
	ASD = 2
	ASA = 15
	ZSD = 3
	ZSA = 7
	XPR = 7
	Delay_spread = 100
####################################################################增加自主添加CDL_Table_功能，优先级低###################################

#######################################################################################################################################
K = 10 ** (XPR / 10)  ##存储
clusterNum = power.shape[0] ##存储
power = 10**(power/10)/(sum(10**(power/10)))   ##存储
PolariedAntennaModel = 'Model_2'  # 存储
##########################################################信道模型算法开关#####################################################
algorithm_switch = pands.read_csv('./algorithm_switch.csv')   #暂时从根目录的excel中导入模型基本参数

Largescale_switch = algorithm_switch['Largescale_switch'].loc[0].lower()
Delay_shift_switch = algorithm_switch['Delay_shift_switch'].loc[0].lower()
CDLtable_shift_switch = algorithm_switch['CDLtable_shift_switch'].loc[0].lower()
Massive_ray_switch= algorithm_switch['Massive_ray_switch'].loc[0].lower()
Cluster_selection_switch= algorithm_switch['Cluster_selection_switch'].loc[0].lower()
Cellspecific_train_switch= algorithm_switch['Cellspecific_train_switch'].loc[0].lower()






