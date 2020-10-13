# # 	脚本名称：CDL.py
# # 	主要功能：根据待配参数需求生成基于38.901的CDL信道时域冲激响应
# # 	主要待配参数：

# #		Channel_type             - CDL信道类型（CDL_C,CDL_D)
# # 	S ;                      -信道模拟器输入振子数
# #   	U ;                      -信道模拟器输出振子数
# #   	Mp;                      -BS垂直同向振子个数；
# #   	Np;                      -BS水平同向振子个数；
# #   	Mpu                      -Ms垂直同向振子个数；
# #   	Npu                      -Ms水平同向振子个数；
# #   	Md=n*WaveLength;         -垂直振子间距；
# #   	Nd=n*WaveLength;         -水平振子间距；
# # 	UEPolarizationPlacement  - UE极化方式
# # 	BSPolarizationPlacement  - BS极化方式
# # 	Am                       -垂直合路参数
# # 	theta_Beam               -电下倾角
# # 	CentralFrequence         - 载波频率
# # 	PolariedAntennaModel     - 极化模型38.901中的模型1和模型2
# #   	period                   -采样周期
# #   	P_num                    -总采样点
# # 	所求参数：H(Rx,Tx,cluster,t)

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
#import matplotlib.pyplot as plt
from Modules import asc_Hdataoutput
from Modules import asc_Vdataoutput
from Modules import asc_dataoutput
from Modules import bingenerate
from Modules import bingenerate_V
from Modules import bingenerate_H
from Channelverification import Channel_verification
################### 基础参数集 #############################################

Main_parameters = pands.read_csv('./CDL_parameters_example.csv')   #从excel中导入模型基本参数
channel_type = Main_parameters['Channeltype'].loc[0].upper()
CentralFrequence = Main_parameters['CentralFrequence'].loc[0]
Cluster_selected = Main_parameters['Cluster_selected'].loc[0]
V_km = Main_parameters['V_km'].loc[0]
LightSpeed = 2.99792458*10**8 				# speed of light
WaveLength = LightSpeed / CentralFrequence
k_CONST = 2*np.pi/WaveLength
V = V_km / 3.6  								# velocity m/s
V_direction = V*np.mat((0, 1, 0))            # UE motion Vector

S = Main_parameters['S'].loc[0]  									# CE input antenna element number
U = Main_parameters['U'].loc[0]  									# CE output antenna element number
UEPolarizationPlacement = Main_parameters['UEPolarizationPlacement'].loc[0]	 # Vertical Cross +；Cross X；Pure Vertical |；Pure Horizontal —
BSPolarizationPlacement = Main_parameters['BSPolarizationPlacement'].loc[0] 			# Vertical Cross +；Cross X；Pure Vertical |；Pure Horizontal —
PolariedAntennaModel = 'Model_2'  # Polarization Type option: 38.901 model_1 or model 2; Used to calculate BS Pattern and F_Tx
Phase_random_up = 'OFF'
############################################信道模拟器仪表参数区#################################################
Emulator_type = Main_parameters['Emulator_type'].loc[0].lower()  #导入信道模拟器类型
if Emulator_type == 'f64':
    SD = 4     # F64采样密度参数，详情见仪表使用说明
    CIR_update_Rate = 2 * SD * V * CentralFrequence / LightSpeed  # F64 CIR 采样率
    period = 1 / CIR_update_Rate
    P_num = Main_parameters['CIR_num'].loc[0]  #导入CIR样点参数
elif Emulator_type == 'ksw':
    SD = 8
    Ts = 2 * SD * V * CentralFrequence / LightSpeed  #KSW 默认采样率，其中采样目的暂时固定位16
    period = 1/Ts
    P_num = 32 * Main_parameters['KSW_cir_num_N'].loc[0]						# 导入系数n，CIR样点数必须为2的n次方 TAG！
else:    #错误仪表标识采用默认的f64配置
    SD = 4
    CIR_update_Rate = 2 * SD * V * CentralFrequence / LightSpeed  # F64 CIR 采样率
    period = 1 / CIR_update_Rate
    P_num = Main_parameters['CIR_num'].loc[0]  #导入CIR样点参数
#####################振子排列##############################

Mp = Main_parameters['Mp'].loc[0] 										# BS垂直同向振子个数；
Np = Main_parameters['Np'].loc[0]										# BS水平同向振子个数；
Mpu = Main_parameters['Mpu'].loc[0]										# #Ms垂直同向振子个数；
Npu = Main_parameters['Npu'].loc[0]										# #Ms水平同向振子个数；
Md = Main_parameters['BS_D_v_lambda'].loc[0]*WaveLength							# #BS垂直振子间距；
Nd = Main_parameters['BS_D_h_lambda'].loc[0]*WaveLength							# #BS平振子间距；
Mdu = Main_parameters['UE_D_v_lambda'].loc[0]*WaveLength							# #UE垂直振子间距；
Ndu = Main_parameters['UE_D_h_lambda'].loc[0]*WaveLength							# #UE水平振子间距；
#####################振子天线方向图参数####################

SLA_v = 30									#dB
A_max = 30									#dB
theta_3dB = 65								#degree
phi_3dB = 65									#degree
theta_Beam = 90								#degree
# LCSTd_theta=0;								#LCS->GCS 暂时未使用；
# LCSTd_phi=0;								#LCS->GCS 暂时未使用；
Am = Main_parameters['Am'].loc[0]										#振子->通道合路数；
Theta_tilt = 90								#电垂直维倾角，90代表无下倾
d_3D = 30									#用于LOS径计算


#####################38.901 CDL 参数集################

if channel_type.lower() == 'CDL_A'.lower() or channel_type.lower() == 'CDL_B'.lower() or channel_type.lower() == 'CDL_C'.lower() : # case  insensitive
	LOS_or_NLOS = 'NLOS'
else:
	LOS_or_NLOS = 'LOS'
	
LOS ='LOS'
rayNum = 20

if channel_type.lower() == 'CDL_A'.lower():
    power = np.array([-13.4, 0, -2.2, -4, -6, -8.2, -9.9, -10.5, -7.5, -15.9, -6.6, -16.7, -12.4, -15.2, -10.8, -11.3, -12.7, -16.2 ,-18.3, -18.9, -16.6, -19.9, -29.7])
    delay = np.array([0.0, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708, 0.5750, 0.7618, 1.5375, 1.8978, 2.2242, 2.1718, 2.4942, 2.5119, 3.0582, 4.0810, 4.4579, 4.5695, 4.7966, 5.0066, 5.3043, 9.6586])
    AOD = np.array([-178.1, -4.2, -4.2, -4.2, 90.2, 90.2, 90.2, 121.5, -81.7, 158.4, -83, 134.8, -153, -172, -129.9, -136, 165.4, 148.4, 132.7, -118.6,  -154.1, 126.5, -56.2 ])
    AOA= np.array([51.3, -152.7,-152.7, -152.7, 76.6, 76.6 , 76.6, -1.8, -41.9, 94.2, 51.9, -115.9, 26.6,  76.6, -7, -23, -47.2, 110.4, 144.5, 155.3, 102, -151.8, 55.2 ])
    ZOD = np.array([50.2, 93.2, 93.2, 93.2, 122, 122, 122, 150.2, 55.2,  26.4, 126.4, 171.6, 151.4, 157.2, 47.2, 40.4, 43.3, 161.8, 10.8, 16.7, 171.7, 22.7, 144.9])
    ZOA = np.array([125.4, 91.3, 91.3, 91.3, 94, 94, 94, 47.1, 56, 30.1, 58.8, 26, 49.2, 143.1, 117.4, 122.7, 123.2, 32.6, 27.2, 15.2, 146, 150.7, 156.1])
    ASD = 5
    ASA = 11
    ZSD = 3
    ZSA = 3
    XPR = 10
    Delay_spread = 100

elif channel_type.lower() == 'CDL_B'.lower():
    power = np.array([0, -2.2,-4,-3.2,-9.8,-1.2,-3.4,-5.2,-7.6,-3,-8.9,-9,-4.8,-5.7,-7.5,-1.9,-7.6,-12.2,-9.8,-11.4,-14.9,-9.2,-11.3])
    delay = np.array([0,0.1072,0.2155,0.2095,0.2870,0.2986,0.3752,0.5055,0.3681,0.3697,0.5700,0.5283,1.1021,1.2756,1.5474,1.7842,2.0169,2.8294,3.0219,3.6187,4.1067,4.2790,4.7834])
    AOD = np.array([9.3,9.3,9.3,-34.1,-65.4,-11.4,-11.4,-11.4,-67.2,52.5,-72,74.3,-52.2,-50.5, 61.4,30.6, -72.5,-90.6,-77.6,-82.6,-103.6,75.6,-77.6])
    AOA= np.array([-173.3,-173.3,-173.3,125.5,-88,155.1,155.1,155.1,-89.8,132.1,-83.6,95.3, 103.7,-87.8,-92.5,-139.1,-90.6,58.6,-79.0,65.8,52.7,88.7,-60.4])
    ZOD = np.array([105.8,105.8,105.8,115.3,119.3,103.2,103.2,103.2,118.2,102.0,100.4, 98.3, 103.4, 102.5,101.4,103,100,115.2,100.5,119.6,118.7,117.8,115.7])
    ZOA = np.array([78.9,78.9,78.9,63.3,59.9,67.5,67.5,67.5,82.6,66.3,61.6,58.0,78.2,82.0, 62.4,78.0,60.9,82.9,60.8,57.3,59.9,60.1,62.3])
    ASD = 10
    ASA = 22
    ZSD = 3
    ZSA = 7
    XPR = 8
    Delay_spread = 100

elif channel_type.lower() == 'CDL_C'.lower(): 
    power = np.array([-4.4,-1.2,-3.5,-5.2,-2.5,0,-2.2,-3.9,-7.4,-7.1,-10.7,-11.1,-5.1,-6.8,-8.7,-13.2,-13.9,-13.9,-15.8,-17.1,-16,-15.7,-21.6,-22.8])
    delay = np.array([0,0.2099,0.2219,0.2329,0.2176,0.6366,0.6448,0.6560,0.6584,0.7935,0.8213,0.9336,1.2285,1.3083,2.1704,2.7105,4.2589,4.6003,5.4902,5.6077,6.3065,6.6374,7.0427,8.6523])
    AOD = np.array([-46.6,-22.8,-22.8,-22.8,-40.7,0.3,0.3,0.3,73.1,-64.5,80.2,-97.1,-55.3,-64.3,-78.5,102.7,99.2,88.8,-101.9,92.2,93.3,106.6,119.5,-123.8])
    AOA= np.array([-101,120,120,120,-127.5,170.4,170.4,170.4,55.4,66.5,-48.1,46.9,68.1,-68.7,81.5,30.7,-16.4,3.8,-13.7,9.7,5.6,0.7,-21.9,33.6])
    ZOD = np.array([97.2,98.6,98.6,98.6,100.6,99.2,99.2,99.2,105.2,95.3,106.1,93.5,103.7,104.2,93.0,104.2,94.9,93.1,92.2,106.7,93.0,92.9,105.2,107.8])
    ZOA = np.array([87.6,72.1,72.1,72.1,70.1,75.3,75.3,75.3,67.4,63.8,71.4,60.5,90.6,60.1,61.0,100.7,62.3,66.7,52.9,61.8,51.9,61.7,58,57])
    ASD = 2
    ASA = 15
    ZSD = 3
    ZSA = 7
    XPR = 7
    Delay_spread = 100

elif channel_type.lower() == 'CDL_D'.lower():

    power = np.array([-0.2,-13.5,-18.8,-21,-22.8,-17.9,-20.1,-21.9,-22.9,-27.8,-23.6,-24.8,-30.0,-27.7])
    delay = np.array([0,0,0.035,0.612,1.363,1.405,1.804,2.596,1.775,4.042,7.937,9.424,9.708,12.525])
    AOD = np.array([0,0,89.2,89.2,89.2,13,13,13,34.6,-64.5,-32.9,52.6,-132.1,77.2])
    AOA = np.array([-180,-180,89.2,89.2,89.2,163,163,163,-137,74.5,127.7,-119.6,-9.1,-83.8])
    ZOD = np.array([98.5,98.5,85.5,85.5,85.5,97.5,97.5,97.5,98.5,88.4,91.3,103.8,80.3,86.5])
    ZOA = np.array([81.5,81.5,86.9,86.9,86.9,79.4,79.4,79.4,78.2,73.6,78.3,87,70.6,72.9])
    ASD = 5
    ASA = 8
    ZSD = 3
    ZSA = 3
    XPR = 11
    Delay_spread = 100
	
elif channel_type.lower() == 'CDL_E'.lower():

    power = np.array([-0.03,-22.03, -15.8, -18.1, -19.8,-22.9, -22.4, -18.6, -20.8, -22.6, -22.3, -25.6, -20.2, -29.8, -29.2])
    delay = np.array([0,0, 0.5133,0.5440, 0.5630,0.5440,0.7112,1.9092,1.9293,1.9589,2.6426,3.7136, 5.4524,12.0034, 20.6419])
    AOD = np.array([0,0,57.5, 57.5,  57.5, -20.1, 16.2, 9.3, 9.3, 9.3, 19, 32.7, 0.5, 55.9, 57.6])
    AOA = np.array([-180,-180,18.2, 18.2, 18.2, 101.8, 112.9, -155.5, -155.5, -155.5, -143.3, -94.7, 147, -36.2, -26])
    ZOD = np.array([99.6, 99.6, 104.2, 104.2, 104.2, 99.4, 100.8, 98.8, 98.8, 98.8, 100.8, 96.4, 98.9, 95.6, 104.6])
    ZOA = np.array([80.4, 80.4, 80.4, 80.4, 80.4, 80.8, 86.3, 82.7, 82.7, 82.7, 82.9,88,81, 88.6, 78.3])
    ASD = 5
    ASA = 11
    ZSD = 3
    ZSA = 7
    XPR = 8
    Delay_spread = 100

else:
	power = np.array([-4.4, -1.2, -3.5, -5.2, -2.5, 0, -2.2, -3.9, -7.4, -7.1, -10.7, -11.1, -5.1, -6.8, -8.7, -13.2, -13.9, -13.9,
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

K = 10**(XPR/10)
clusterNum = power.shape[0]
#power = 10**(power/10)/(sum(10**(power/10)))   # dB to Amplitude

#************************************************* 删除模型中基站无法正常发射接收的簇******************************************
ThitaRange = 42 #垂直角度范围
PhiRange = 120  #水平角度范围

if Cluster_selected.lower() != 'OFF'.lower():
	clusterSet = []
	for clusterIndex in range(0, clusterNum, 1):

		if AOD[clusterIndex] >= -PhiRange / 2 and AOD[clusterIndex] <= PhiRange / 2 and ZOD[
			clusterIndex] >= 90 - ThitaRange / 2 and ZOD[clusterIndex] <= 90 + ThitaRange / 2:
			clusterSet.append(clusterIndex)
	power = power[clusterSet]
	power = 10 ** (power / 10) / (sum(10 ** (power / 10)))  # dB to linear
	delay = delay[clusterSet]
	AOD = AOD[clusterSet]
	AOA = AOA[clusterSet]
	ZOD = ZOD[clusterSet]
	ZOA = ZOA[clusterSet]
	clusterNum = delay.shape[0]
else:
	power = 10 ** (power / 10) / (sum(10 ** (power / 10)))  # dB to linear


#**************************************子径偏转设置*****************************************************************************************


path_delta_deg = np.array((0.0447, 0.1413, 0.2492, 0.3715, 0.5129, 0.6797, 0.8844, 1.1481, 1.5195, 2.1551))
path_delta_deg1 = np.array([path_delta_deg, -1 * path_delta_deg])
path_delta_deg1 = np.asmatrix(path_delta_deg1)
path_delta_deg1 = path_delta_deg1.flatten('F').T

if LOS_or_NLOS.lower() == LOS.lower():
	cluster_start_index = 2
	path_delta_deg = np.tile(path_delta_deg1, (1, clusterNum - 1))
else:
	cluster_start_index = 1
	path_delta_deg = np.tile(path_delta_deg1, (1, clusterNum))

LOS_ZOD = ZOD[0]
LOS_AOD = AOD[0]
LOS_AOA = AOA[0]
LOS_ZOA = ZOA[0]

ZOD = np.tile(ZOD[range(cluster_start_index-1, len(ZOD))], (rayNum, 1)) + ZSD*path_delta_deg
AOD = np.tile(AOD[range(cluster_start_index-1, len(AOD))], (rayNum, 1)) + ASD*path_delta_deg
AOA = np.tile(AOA[range(cluster_start_index-1, len(AOA))], (rayNum, 1)) + ASA*path_delta_deg
ZOA = np.tile(ZOA[range(cluster_start_index-1, len(ZOA))], (rayNum, 1)) + ZSA*path_delta_deg
	
ZOD_d = ZOD
AOD_d = AOD

AOD = np.deg2rad(AOD)
AOA = np.deg2rad(AOA)
ZOD = np.deg2rad(ZOD)
ZOA = np.deg2rad(ZOA)

LOS_AOD = np.deg2rad(LOS_AOD)
LOS_AOA = np.deg2rad(LOS_AOA)
LOS_ZOD = np.deg2rad(LOS_ZOD)
LOS_ZOA = np.deg2rad(LOS_ZOA)


#####################Ray coupling CMCC & DaTang################

MZOD = np.array([range(11, 21), range(1, 11)])-1
MZOD = MZOD.flatten()						# 变换成一维矩阵
MAOD = np.array([range(11, 21), range(1, 11)])-1
MAOD = MAOD.flatten()						# 变换成一维矩阵
MAOA = np.array(range(1, rayNum+1))-1
MZOA = np.array(range(1, rayNum+1))-1
					  							
N = clusterNum
M = rayNum
fd = V/WaveLength

#################### 随机相位 ##########################
'''
if Phase_random_up == 'ON':
	InitialPhase_VV = np.random.rand(U, S, clusterNum, rayNum)*2*np.pi
	InitialPhase_VH = np.random.rand(U, S, clusterNum, rayNum)*2*np.pi
	InitialPhase_HV = np.random.rand(U, S, clusterNum, rayNum)*2*np.pi
	InitialPhase_HH = np.random.rand(U, S, clusterNum, rayNum)*2*np.pi
else:
	InitialPhase_VV = np.random.rand(clusterNum, rayNum) * 2 * np.pi
	InitialPhase_VH = np.random.rand(clusterNum, rayNum) * 2 * np.pi
	InitialPhase_HV = np.random.rand(clusterNum, rayNum) * 2 * np.pi
	InitialPhase_HH = np.random.rand(clusterNum, rayNum) * 2 * np.pi
'''

'''
InitialPhase_VV = np.random.rand(clusterNum, rayNum)*2*np.pi
InitialPhase_VH = np.random.rand(clusterNum, rayNum)*2*np.pi
InitialPhase_HV = np.random.rand(clusterNum, rayNum)*2*np.pi
InitialPhase_HH = np.random.rand(clusterNum, rayNum)*2*np.pi
'''


InitialPhase = scio.loadmat('InitialPhase.mat')
InitialPhase_VV = InitialPhase['InitialPhase_VV']
InitialPhase_VH = InitialPhase['InitialPhase_VH']
InitialPhase_HV = InitialPhase['InitialPhase_HV']
InitialPhase_HH = InitialPhase['InitialPhase_HH']


# InitialPhase_VV = np.random.rand(clusterNum,rayNum)*2*np.pi;
# InitialPhase_VH = np.random.rand(clusterNum,rayNum)*2*np.pi;
# InitialPhase_HV = np.random.rand(clusterNum,rayNum)*2*np.pi;
# InitialPhase_HH = np.random.rand(clusterNum,rayNum)*2*np.pi;

# fvv = open('vv.txt','w');
# fvh = open('vh.txt','w');
# fhv = open('hv.txt','w');
# fhh = open('hh.txt','w');

# np.savetxt(fvv,InitialPhase_VV);
# np.savetxt(fhv,InitialPhase_HV);
# np.savetxt(fvh,InitialPhase_VH);
# np.savetxt(fhh,InitialPhase_HH);


# fvv.close();
# fhv.close();
# fvh.close();
# fhh.close();


# for s in range(0,S):
	# InitialPhase_VV1[s,:,:] =InitialPhase_VV;
	# InitialPhase_VH1[s,:,:] =InitialPhase_VH;
	# InitialPhase_HV1[s,:,:] =InitialPhase_HV;
	# InitialPhase_HH1[s,:,:] =InitialPhase_HH;

# InitialPhase_VV = InitialPhase_VV1;	
# InitialPhase_HV = InitialPhase_HV1;	
# InitialPhase_VH = InitialPhase_VH1;	
# InitialPhase_HH = InitialPhase_HH1;	
	
########################################基站和终端的极化斜角赋值###########################

if UEPolarizationPlacement.lower() == 'UEPureHorizontal'.lower():
	slantA = 90
elif UEPolarizationPlacement.lower() =='UECross'.lower():
	slantA = 45
else:
	slantA = 0
		
if BSPolarizationPlacement.lower() == 'BSPureHorizontal'.lower():
    slantB = 90
elif BSPolarizationPlacement.lower() == 'BSCross'.lower():
    slantB = 45
else:
    slantB = 0

		
#####################计算终端天线阵子坐标################

MsElementPosition = np.zeros((int(U), 3), dtype=float)  # 终端阵子坐标
if UEPolarizationPlacement.lower() == 'UECross'.lower() or UEPolarizationPlacement.lower() == 'UEVerticalCross'.lower():
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
	MsElementPosition[1:U+1:2, :] = location_antennatemp
	location_antennatemp = []

else:
	location_antennatemp = np.array(np.zeros((1, int(U  * 3))), dtype=float)
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


#矩阵形式的阵子坐标生成法，由于存在取整操作，为避免误差影响而被弃用
'''
if UEPolarizationPlacement.lower() == 'UECross'.lower() or UEPolarizationPlacement.lower() == 'UEVerticalCross'.lower():
	MsElementPositiontemp = np.zeros((int(U/2), 3), dtype=float)
	MsElementPositiontemp[:, 2] = np.linspace(0, (Mpu*Npu-1)*Md, (Mpu*Npu)) # start stop number
	MsElementPositiontemp[:, 2] = MsElementPositiontemp[:, 2] % (Mpu*Md)
	MsElementPositiontemp[:, 1] = np.linspace(0, (Mpu*Npu-1)*Md, (Mpu*Npu))
	MsElementPositiontemp[:, 1] = Nd*np.floor(MsElementPositiontemp[:, 1]/(Mpu*Md))
		
	MsElementPosition = np.zeros((int(U), 3), dtype=float)

	MsElementPosition[range(0, U, 2), 1] = MsElementPositiontemp[:, 1]
	MsElementPosition[range(1, U, 2), 1] = MsElementPositiontemp[:, 1]
	MsElementPosition[range(0, U, 2), 2] = MsElementPositiontemp[:, 2]
	MsElementPosition[range(1, U, 2), 2] = MsElementPositiontemp[:, 2]
	
	MsElementPositiontemp=[]

else:
	MsElementPositiontemp = np.zeros((int(U), 3), dtype=float)

	MsElementPositiontemp[:, 2] = np.linspace(0, (Mpu*Npu-1)*Md,(Mpu*Npu))
	MsElementPositiontemp[:, 2] = MsElementPositiontemp[:, 2] % (Mpu*Md)
	MsElementPositiontemp[:, 1] = np.linspace(0, (Mpu*Npu-1)*Md, (Mpu*Npu))
	MsElementPositiontemp[:, 1] = Nd*np.floor(MsElementPositiontemp[:, 1]/(Mpu*Md))
		
	MsElementPosition = np.zeros((int(U), 3), dtype=float)
	MsElementPosition = MsElementPositiontemp
	MsElementPositiontemp=[]
'''
###################计算基站天线阵子坐标##################


BsElementPosition = np.zeros((int(S), 3), dtype=float)    # 基站侧阵子坐标

if BSPolarizationPlacement.lower() == 'BSCross'.lower() or BSPolarizationPlacement.lower() == 'BSVerticalCross'.lower():  #双极化阵子排布
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

#矩阵形式的阵子坐标生成法，由于存在取整操作，为避免误差影响而被弃用
'''
if BSPolarizationPlacement.lower() == 'BSCross'.lower() or BSPolarizationPlacement.lower() == 'BSVerticalCross'.lower():
	BsElementPosition = np.zeros((int(S/2), 3), dtype=float)
	BsElementPosition[:, 2] = np.linspace(0, (Mp*Np-1), Mp*Np)
	BsElementPosition[:, 2] = np.mod(BsElementPosition[:, 2], Mp)*Md
	BsElementPosition[:, 1] = np.linspace(0, (Mp*Np-1)*Md, Mp*Np)
	BsElementPosition[:, 1] = Nd*np.floor(BsElementPosition[:, 1]/(Mp*Md))
	BsElementPosition = np.tile(BsElementPosition, (2, 1))
	
else: 

	BsElementPosition = np.zeros((int(S), 3), dtype=float)
	BsElementPosition[:, 2] = np.linspace(0, (Mp*Np-1), Mp*Np)
	BsElementPosition[:, 2] = np.mod(BsElementPosition[:, 2], Mp)*Md
	BsElementPosition[:, 1] = np.linspace(0, (Mp*Np-1)*Md, Mp*Np)
	BsElementPosition[:, 1] = Nd*np.floor(BsElementPosition[:, 1]/(Mp*Md))
'''





###################规定样点与时间间隔##################
PhaseCompensationMatrix = np.sqrt(1/Mp)*np.exp(-1j*(k_CONST*Md*np.mod(range(S-1, -1,-1), Mp)*np.cos(np.deg2rad(Theta_tilt))))


t = np.array(range(1, P_num+1))*period  		# Time Samples


#######################信道系数生成主体###############################################

H_average = np.zeros((U, int(S/Am), clusterNum, P_num), dtype=complex)  		# Output Target size of (u,s,cluster,package)
H_real_average = np.zeros((U, int(S/Am), clusterNum, P_num), dtype=float)		# real
H_imag_average = np.zeros((U, int(S/Am), clusterNum, P_num), dtype=float) 	# imag

ModelTpye = 'Model_1'

for u in range(0, U):

	print('Calculating: Ms antenna', u)
	du = np.asmatrix(MsElementPosition[u, :])


	if UEPolarizationPlacement.lower() =='UECross'.lower() or UEPolarizationPlacement.lower() == 'UEVerticalCross'.lower():
		if np.mod(u, 2) == 0:
			X_slant_angle = slantA
		else:
			if slantA == 45:
				X_slant_angle = slantA-90
			else:
				X_slant_angle = slantA+90
	else:
		X_slant_angle = slantA
	F_rx = np.mat([np.cos(np.deg2rad(X_slant_angle)),np.sin(np.deg2rad(X_slant_angle))])
	
	for s in range(0, int(S)):

		ds = np.asmatrix(BsElementPosition[s,:])
		
		if BSPolarizationPlacement.lower() == 'BSCross'.lower() or BSPolarizationPlacement.lower() == 'BSVerticalCross'.lower():
			if s < int(S/2):
				XslantA = slantB
			else:
				if slantB == 45:
					XslantA = slantB-90
				else:
					XslantA = slantB+90
		else:
			XslantA = slantB
		for	iCluster in range(0,clusterNum):
			
			if iCluster == 0 and LOS_or_NLOS.lower() == LOS.lower():
				LOS_ZOD_d = np.rad2deg(LOS_ZOD)
				LOS_AOD_d = np.rad2deg(LOS_AOD)

				# A_thetaphiV = -min(12*((LCSTd_theta+LOS_ZOD_d-90)/theta_3dB)**2,SLA_v);
				# A_thetaphiH = -min(12*((LCSTd_phi+LOS_AOD_d)/phi_3dB)**2,A_max);
				A_thetaphiV = -min(12*((LOS_ZOD_d-90)/theta_3dB)**2,SLA_v)
				A_thetaphiH = -min(12*((LOS_AOD_d)/phi_3dB)**2,A_max)
				A_thetaphi = 10**(-min(-(A_thetaphiV+A_thetaphiH),A_max)/10)		# Antenna Gain of Tx
				temp = np.zeros((1, P_num), dtype=float)
				
				if (PolariedAntennaModel.lower() ==ModelTpye.lower()):
					CosPsi = (np.cos(np.deg2rad(XslantA))*np.sin(LOS_ZOD)+  np.sin(np.deg2rad(XslantA))*np.sin(LOS_AOD)*np.cos(LOS_ZOD)  ) / np.sqrt(1-(np.cos(np.deg2rad(XslantA))*np.cos(LOS_ZOD)-np.sin(np.deg2rad(XslantA))*np.sin(LOS_AOD)*np.sin(LOS_ZOD))**2)
					SinPsi = (np.sin(np.deg2rad(XslantA))*np.cos(LOS_AOD))/np.sqrt(1-(np.cos(np.deg2rad(XslantA))*np.cos(LOS_ZOD)-np.sin(np.deg2rad(XslantA))*np.sin(LOS_AOD)*np.sin(LOS_ZOD))**2);
					F_tx=(np.mat([[CosPsi,-SinPsi],[SinPsi,CosPsi]])*np.mat([ np.sqrt(A_thetaphi),0]).H).H
				else:
					F_tx=np.mat([np.sqrt(A_thetaphi)*np.cos(np.deg2rad(XslantA)),np.sqrt(A_thetaphi)*np.sin(np.deg2rad(XslantA))])
			
				tx = np.mat([np.sin(LOS_ZOD)*np.cos(LOS_AOD),np.sin(LOS_ZOD)*np.sin(LOS_AOD),np.cos(LOS_ZOD)]).T # tx unit vector
				rx = np.mat([np.sin(LOS_ZOA)*np.cos(LOS_AOA),np.sin(LOS_ZOA)*np.sin(LOS_AOA),np.cos(LOS_ZOA)]).T # rx unit vector

				temp = np.sqrt(power[iCluster])*F_rx * np.mat([[1,0],[0,-1]])*F_tx.T * np.exp(-1j*k_CONST*d_3D)*PhaseCompensationMatrix[s]*np.exp(1j*k_CONST*rx.T*du.T)*np.exp(1j*k_CONST*tx.T*ds.T)*np.exp(1j*(k_CONST*rx.T*V_direction.T*t))

			else:
				temp = np.asmatrix(np.zeros((M, P_num), dtype=complex))
				if LOS_or_NLOS == LOS:
					iClusterIndex = iCluster -1
				else:
					iClusterIndex = iCluster
					
				PhaseCompensation = PhaseCompensationMatrix[s]
				poweri = power[iCluster]
				
				for isubpath in range(0, M):

					# A_thetaphiV = -min(12*((LCSTd_theta+ZOD_d[int(np.nonzero(MZOD == isubpath)[0]),iClusterIndex]-90)/theta_3dB)**2,SLA_v);
					# A_thetaphiH = -min(12*((LCSTd_phi+AOD_d[int(np.nonzero(MAOD == isubpath)[0]),iClusterIndex])/phi_3dB)**2,A_max);
					A_thetaphiV = -min(12*((ZOD_d[int(np.nonzero(MZOD == isubpath)[0]),iClusterIndex]-90)/theta_3dB)**2,SLA_v)
					A_thetaphiH = -min(12*((AOD_d[int(np.nonzero(MAOD == isubpath)[0]),iClusterIndex])/phi_3dB)**2,A_max)
					A_thetaphi = 10**(-min(-(A_thetaphiV+A_thetaphiH),A_max)/10)

					if (PolariedAntennaModel.lower() ==ModelTpye.lower()):
						CosPsi = (np.cos(np.deg2rad(XslantA))*np.sin(ZOD[int(np.nonzero(MZOD == isubpath)[0]),iClusterIndex])+ np.sin(np.deg2rad(XslantA))*np.sin(AOD[int(np.nonzero(MAOD == isubpath)[0]),iClusterIndex])*np.cos(ZOD[int(np.nonzero(MZOD == isubpath)[0]),iClusterIndex])  ) / np.sqrt(1-(np.cos(np.deg2rad(XslantA))*np.cos(ZOD[int(np.nonzero(MZOD == isubpath)[0]),iClusterIndex])-np.sin(np.deg2rad(XslantA))*np.sin(AOD[int(np.nonzero(MAOD == isubpath)[0]),iClusterIndex])*np.sin(ZOD[int(np.nonzero(MZOD == isubpath)[0]),iClusterIndex]))**2)
						SinPsi = (np.sin(np.deg2rad(XslantA))*np.cos(AOD[int(np.nonzero(MAOD == isubpath)[0]),iClusterIndex]))/np.sqrt(1-(np.cos(np.deg2rad(XslantA))*np.cos(ZOD[int(np.nonzero(MZOD == isubpath)[0]),iClusterIndex])-np.sin(np.deg2rad(XslantA))*np.sin(AOD[int(np.nonzero(MAOD == isubpath)[0]),iClusterIndex])*np.sin(ZOD[int(np.nonzero(MZOD == isubpath)[0]),iClusterIndex]))**2)
						F_tx=(np.mat([[CosPsi, -SinPsi], [SinPsi, CosPsi]])*np.mat([np.sqrt(A_thetaphi),0]).H).H
					else:
						F_tx=np.mat([np.sqrt(A_thetaphi)*np.cos(np.deg2rad(XslantA)), np.sqrt(A_thetaphi)*np.sin(np.deg2rad(XslantA))])
				
					tx =  np.mat([np.sin(ZOD[int(np.nonzero(MZOD == isubpath)[0]), iClusterIndex])*np.cos(AOD[int(np.nonzero(MAOD == isubpath)[0]),iClusterIndex]),np.sin(ZOD[int(np.nonzero(MZOD == isubpath)[0]),iClusterIndex])*np.sin(AOD[int(np.nonzero(MAOD == isubpath)[0]),iClusterIndex]),np.cos(ZOD[int(np.nonzero(MZOD == isubpath)[0]),iClusterIndex])]).T
					rx =  np.mat([np.sin(ZOA[int(np.nonzero(MZOA == isubpath)[0]), iClusterIndex])*np.cos(AOA[int(np.nonzero(MAOA == isubpath)[0]),iClusterIndex]),np.sin(ZOA[int(np.nonzero(MZOA == isubpath)[0]),iClusterIndex])*np.sin(AOA[int(np.nonzero(MAOA == isubpath)[0]),iClusterIndex]),np.cos(ZOA[int(np.nonzero(MZOA == isubpath)[0]),iClusterIndex])]).T

					if Phase_random_up == 'ON':
						temp[isubpath, :] = np.sqrt(poweri/rayNum)*F_rx* np.mat([[np.exp(1j*InitialPhase_VV[u, s, iClusterIndex, isubpath]),np.sqrt(1/K)*PhaseCompensation*np.exp(1j*InitialPhase_VH[u, s, iClusterIndex, isubpath])],\
						[np.sqrt(1/K)*np.exp(1j*InitialPhase_HV[u, s, iClusterIndex, isubpath]), np.exp(1j*InitialPhase_HH[u, s, iClusterIndex, isubpath])]])*\
						F_tx.H*PhaseCompensationMatrix[s]*np.exp(1j*k_CONST*rx.T*du.T)*np.exp(1j*k_CONST*tx.T*ds.T)*np.exp(1j*k_CONST*rx.T*V_direction.T*t)
					else:
						temp[isubpath, :] = np.sqrt(poweri/rayNum)*F_rx* np.mat([[np.exp(1j*InitialPhase_VV[iClusterIndex, isubpath]),np.sqrt(1/K)*np.exp(1j*InitialPhase_VH[iClusterIndex, isubpath])],\
						[np.sqrt(1/K)*np.exp(1j*InitialPhase_HV[iClusterIndex, isubpath]), np.exp(1j*InitialPhase_HH[iClusterIndex, isubpath])]])*\
						F_tx.H*PhaseCompensationMatrix[s]*np.exp(1j*k_CONST*rx.T*du.T)*np.exp(1j*k_CONST*tx.T*ds.T)*np.exp(1j*k_CONST*rx.T*V_direction.T*t)


				temp = temp.sum(axis=0) 		# sum of different rays to produce waveform of one cluster

			H_average[u, int(np.floor(s/Am)), iCluster, :] = (np.squeeze(H_average[u, int(np.floor(s/Am)), iCluster, :]).T+ temp) # 振子合并

			
			if np.mod(s+1, Am)==0:
				H_real_average[u, int(np.floor(s/Am)), iCluster, :] = np.real(np.squeeze(H_average[u, int(np.floor(s/Am)), iCluster,:]))
				H_imag_average[u, int(np.floor(s/Am)), iCluster, :] = np.imag(np.squeeze(H_average[u, int(np.floor(s/Am)), iCluster,:]))
# loop U				
	# loop S
		# loop Cluster





##########################信道关键特征验证#######################

#Channel_verification(delay, period, H_average, LOS_or_NLOS, channel_type, Mp, Np, Am)


###########################根据实际的连线方式进行端口映射，重点针对32通道进行处理######################################
H_V_average = np.zeros((U, int(S/Am/2), clusterNum, P_num), dtype=complex)
H_H_average = np.zeros((U, int(S/Am/2), clusterNum, P_num), dtype=complex)
H_V_average=H_average[:, 0:int(S/Am/2), :, :]
H_H_average=H_average[:, int(S/Am/2):int(S/Am), :, :]
H_up_average = np.zeros((U, int(S/Am/2), clusterNum, P_num), dtype=complex)
H_down_average = np.zeros((U, int(S/Am/2), clusterNum, P_num), dtype=complex)
H_up_average[:, 0:int(S/Am/2)-1:2, :, :] = H_V_average[:, 0:int(S/Am/2)-1:2, :, :]
H_up_average[:, 1:int(S/Am/2):2, :, :] = H_H_average[:, 0:int(S/Am/2)-1:2, :, :]
H_down_average[:, 0:int(S/Am/2)-1:2, :, :] = H_V_average[:, 1:int(S/Am/2):2, :, :]
H_down_average[:, 1:int(S/Am/2):2, :, :] = H_H_average[:, 1:int(S/Am/2):2, :, :]
H_average[:, 0:int(S/Am/2), :, :] = H_up_average
H_average[:, int(S/Am/2):int(S/Am), :, :] = H_down_average

scio.savemat('ChannelData.mat', {'H_average': H_average})

delay_for_save = delay*Delay_spread
scio.savemat('delay.mat', {'delay': delay_for_save})
#######################根据信道模拟器类型进行信道数据打印####################


if Emulator_type == 'f64':
	if S/Am > 32:
		print(asc_Vdataoutput(P_num, CentralFrequence, SD, CIR_update_Rate, S/Am, U, 'ASC_OutputResults45', channel_type, V_km, delay, H_average, H_imag_average, H_real_average, Delay_spread))
		print(asc_Hdataoutput(P_num, CentralFrequence, SD, CIR_update_Rate, S/Am, U, 'ASC_OutputResults_45', channel_type, V_km, delay, H_average, H_imag_average, H_real_average, Delay_spread))
	else:
		print(asc_dataoutput(P_num, CentralFrequence, SD, CIR_update_Rate, S / Am, U, 'ASC_OutputResults', channel_type,V_km, delay, H_average, H_imag_average, H_real_average, Delay_spread))
else:
	if S / Am > 32:
		print(bingenerate_V('Bin_OutputResults_V', H_average))
		print(bingenerate_H('Bin_OutputResults_H', H_average))
		power_sum = np.squeeze((abs(np.squeeze((H_average.sum(axis=2))))**2).sum(axis=2)) / P_num
		power_average_xml = sum(sum(power_sum))/ (S/Am) / U
		power_max_xml = np.max(abs(H_average)** 2)
		print(power_average_xml)
		print(power_max_xml)

	else:
		print(bingenerate('Bin_OutputResults', H_average))
		power_sum = np.squeeze((abs(np.squeeze((H_average.sum(axis=2))))**2).sum(axis=2)) / P_num
		power_average_xml = sum(sum(power_sum))/ (S/Am) / U
		power_max_xml = np.max(abs(H_average)** 2)
		print(power_average_xml)
		print(power_max_xml)


	






