
import sys
import re
import os
import math
import numpy as np
from numpy import ceil
import scipy as sp
import datetime 
from sys import argv
import matplotlib.pyplot as plt

# F64 ASC文件生成，分为1台F64双极化，两台F64分别单极化
def	asc_dataoutput(PcaketNum, Carrier_Frequency, Sample_Density, CIR_Update_Rate, S, U, floderName, channel_type, V, delay, H_average, H_imag_average, H_real_average, delay_spread):

	tmp ='asc_FINISHED'

	nCluster = len(delay)
	Delay_Resolution = 0
	variablevalue =	np.array([PcaketNum, nCluster, Carrier_Frequency, 00, 00, 00, Delay_Resolution, Sample_Density, CIR_Update_Rate])
	variablename = ['CIRs','Taps/CIR','Carrier_Frequency',\
    'Route_Closed','CIRUpdateRate_Unlocked','CarrierFrequency_Unlocked',\
    'Delay_Resolution','Sample_Density','CIR_Update_Rate']

	delay_all = np.tile(delay,(PcaketNum, 1))
	delay_all = np.round(delay_all*delay_spread, 3)
	H_real_average = np.round(H_real_average, 7)
	H_imag_average = np.round(H_imag_average, 7)
	if not os.path.exists(floderName):
		os.mkdir(floderName)  

	for u in range(0, U):
           # print('Print UE Antenna:', u,'to asc file');
		for s in range(0, S):
			filename = '{3}/DU_asc_dataoutput_{4}_{0}km_{1}_{2}.asc'.format(int(V),u+1,s+1,floderName,channel_type.upper())
			fw = open(filename, 'w')
			header ='*****  Header   *****'+'\n'
			fw.write(header)
			for i in range(0, 9):
				if (i != 5) and (i != 4) and(i != 3) and (i !=8):
					fw.write(str(variablevalue[i])+'\t')

				elif i == 8:
					fw.write(format(variablevalue[i],'.4f')+'\t')
				fw.write(variablename[i]+'\n')
			rearer = '*****  Tap data *****'
			fw.write(rearer+'\n')
			title = 'Delay	Re	Im	'*nCluster
			fw.write(title+'\n')
			
			for i in range(0, PcaketNum):
				line = ''
				for ii in range(0, nCluster):
					line += str(delay_all[i, ii])+'\t'+ str(H_real_average[u, s, ii, i])+'\t' + str(H_imag_average[u, s, ii, i]) +'\t'
				line = line + '\n'
				fw.write(line)
			fw.close
			
	return tmp
	
	
	
def	asc_Vdataoutput(PcaketNum, Carrier_Frequency, Sample_Density, CIR_Update_Rate, S, U, floderName, channel_type, V, delay, H_average, H_imag_average, H_real_average, delay_spread):

	tmp ='V_asc_FINISHED'
	nCluster = len(delay)
	Delay_Resolution = 0
	variablevalue =	np.array([PcaketNum, nCluster, Carrier_Frequency, 00, 00, 00, Delay_Resolution, Sample_Density, CIR_Update_Rate])
	variablename = ['CIRs','Taps/CIR','Carrier_Frequency',\
    'Route_Closed','CIRUpdateRate_Unlocked','CarrierFrequency_Unlocked',\
    'Delay_Resolution','Sample_Density','CIR_Update_Rate']

	delay_all = np.tile(delay, (PcaketNum, 1))
	delay_all = np.round(delay_all*delay_spread, 3)
	H_real_average = np.round(H_real_average, 7)
	H_imag_average = np.round(H_imag_average, 7)
	if not os.path.exists(floderName):
		os.mkdir(floderName)

	for u in range(0, U):
		for s in range(0,int(S/2)):
			#print('Print UE Antenna:', u, 'BS Antenna:', s, 'to asc file')
			filename = '{3}/DU_asc_dataoutput_{4}_{0}km_{1}_{2}.asc'.format(int(V),u+1,s+1,floderName,channel_type.upper())
			fw = open(filename,'w')
			header ='*****  Header   *****'+'\n'
			fw.write(header)
			for i in range(0, 9):
				if (i != 5) and (i != 4) and (i !=3) and (i !=8):
					fw.write(str(variablevalue[i])+'\t')

				elif i == 8:
					fw.write(format(variablevalue[i], '.4f')+'\t')
				fw.write(variablename[i]+'\n')
			rearer = '*****  Tap data *****'
			fw.write(rearer+'\n')
			# if channel_type.lower()=='CDL_C'.lower():
				# title = 'Delay	Re	Im	Delay	Re	Im	Delay	Re	Im	Delay	Re	Im	Delay	Re	Im	Delay	Re	Im	Delay	Re	Im	Delay	Re	Im	Delay	Re	Im  Delay	Re	Im  Delay	Re	Im  Delay	Re	Im  Delay	Re	Im  Delay	Re	Im  Delay	Re	Im  Delay	Re	Im  Delay	Re	Im  Delay	Re	Im  Delay	Re	Im  Delay	Re	Im  Delay	Re	Im  Delay	Re	Im  Delay	Re	Im  Delay	Re	Im' ;
			# else:
				# title = 'Delay	Re	Im	Delay	Re	Im	Delay	Re	Im	Delay	Re	Im	Delay	Re	Im	Delay	Re	Im	Delay	Re	Im	Delay	Re	Im  Delay	Re	Im  Delay	Re	Im  Delay	Re	Im  Delay	Re	Im  Delay	Re	Im  Delay	Re	Im' ;
			title = 'Delay	Re	Im	'*nCluster
			fw.write(title+'\n')
			# print('AA:',datetime.datetime.now())

			# delay_all = np.round(delay_all*100,3);
			# H_real_average =  np.round(H_real_average,6);
			# H_imag_average = np.round(H_imag_average,6);
			for i in range(0, PcaketNum):
				line = ''
				for ii in range(0, nCluster):
					line += str(delay_all[i, ii])+'\t' + str(H_real_average[u,s,ii,i])+'\t' + str(H_imag_average[u, s, ii, i]) +'\t'
				line = line + '\n'
				fw.write(line)
			fw.close
			# print('BB:',datetime.datetime.now())
			
	return tmp
	
	
def	asc_Hdataoutput(PcaketNum, Carrier_Frequency, Sample_Density, CIR_Update_Rate, S, U, floderName, channel_type, V, delay, H_average, H_imag_average, H_real_average, delay_spread):

	tmp ='H_asc_FINISHED'
	nCluster = len(delay)
	Delay_Resolution = 0
	variablevalue =	np.array([PcaketNum, nCluster, Carrier_Frequency, 00, 00, 00, Delay_Resolution, Sample_Density, CIR_Update_Rate])
	variablename = ['CIRs','Taps/CIR','Carrier_Frequency',\
    'Route_Closed','CIRUpdateRate_Unlocked','CarrierFrequency_Unlocked',\
    'Delay_Resolution','Sample_Density','CIR_Update_Rate']

	delay_all = np.tile(delay,(PcaketNum,1))
	delay_all = np.round(delay_all*delay_spread,3)
	H_real_average = np.round(H_real_average,7)
	H_imag_average = np.round(H_imag_average,7)
	if not os.path.exists(floderName):
		os.mkdir(floderName)  

	for u in range(0, U):
		for s in range(int(S/2),int(S)):
			#print('Print UE Antenna:', u, 'BS Antenna:', s, 'to asc file')
			filename = '{3}/DU_asc_dataoutput_{4}_{0}km_{1}_{2}.asc'.format(int(V),u+1,s-31,floderName,channel_type.upper())
			fw = open(filename, 'w')
			header ='*****  Header   *****'+'\n'
			fw.write(header)
			for i in range(0, 9):
				if (i != 5) and (i != 4) and (i != 3) and (i != 8):
					fw.write(str(variablevalue[i])+'\t')

				elif i == 8:
					fw.write(format(variablevalue[i], '.4f')+'\t')
				fw.write(variablename[i]+'\n')
			rearer = '*****  Tap data *****'
			fw.write(rearer+'\n')
			title = 'Delay	Re	Im	'*nCluster
			fw.write(title+'\n')
			for i in range(0, PcaketNum):
				line = ''
				for ii in range(0, nCluster):
					line += str(delay_all[i, ii])+'\t' + str(H_real_average[u, s, ii, i])+'\t' + str(H_imag_average[u, s, ii, i]) + '\t'
				line = line + '\n'
				fw.write(line)
			fw.close
			
	return tmp

# KSW S02 BIN 文件生成
def	bingenerate(flodername, h_average):
	tmp = 'Bin_FINISHED'
	u = np.size(h_average, 0)
	s = np.size(h_average, 1)
	cluster_num = np.size(h_average, 2)
	cir_num = np.size(h_average, 3)
	h_temp = np.zeros((u, s*cluster_num*cir_num), dtype=complex)
	datatemp = np.zeros((2*s*cluster_num*cir_num, 1), dtype=float)
	if not os.path.exists(flodername):
		os.mkdir(flodername)
	for ii_index in range(0, u):
		h_temp[ii_index, :] = np.squeeze(h_average[ii_index, :, :, :]).reshape((1, s*cluster_num*cir_num), order='F')
		datatemp[0:np.size(datatemp, 0):2, :] = np.transpose(np.real(h_temp[ii_index, :][np.newaxis, :]))
		datatemp[1:np.size(datatemp, 0):2, :] = np.transpose(np.imag(h_temp[ii_index, :][np.newaxis, :]))
		datatemp = np.float32(datatemp)
		filename = '{1}/default_BS{2}_UE{3}_Ant{0}.bin'.format(ii_index, flodername, 0, 0)
		fw = open(filename, 'wb')
		fw.write(datatemp)
		fw.close
	return tmp

# KSW S02 BIN 文件生成
def	bingenerate_V(flodername, h_average):
	tmp = 'Bin_V_FINISHED'
	u = np.size(h_average, 0)
	s = np.size(h_average, 1)
	cluster_num = np.size(h_average, 2)
	cir_num = np.size(h_average, 3)
	h_temp = np.zeros((u, int(s/2)*cluster_num*cir_num), dtype=complex)
	datatemp = np.zeros((s*cluster_num*cir_num, 1), dtype=float)
	if not os.path.exists(flodername):
		os.mkdir(flodername)
	for ii_index in range(0, u):
		h_temp[ii_index, :] = np.squeeze(h_average[ii_index, 0:int(s/2), :, :]).reshape((1, int(s/2)*cluster_num*cir_num), order='F')
		datatemp[0:np.size(datatemp, 0):2, :] = np.transpose(np.real(h_temp[ii_index, :][np.newaxis, :]))
		datatemp[1:np.size(datatemp, 0):2, :] = np.transpose(np.imag(h_temp[ii_index, :][np.newaxis, :]))
		datatemp = np.float32(datatemp)
		filename = '{1}/default_BS{2}_UE{3}_Ant{0}.bin'.format(ii_index, flodername, 0, 0)
		fw = open(filename, 'wb')
		fw.write(datatemp)
		fw.close
	return tmp

def	bingenerate_H(flodername, h_average):
	tmp = 'Bin_H_FINISHED'
	u = np.size(h_average, 0)
	s = np.size(h_average, 1)
	cluster_num = np.size(h_average, 2)
	cir_num = np.size(h_average, 3)
	h_temp = np.zeros((u, int(s/2)*cluster_num*cir_num), dtype=complex)
	datatemp = np.zeros((s*cluster_num*cir_num, 1), dtype=float)
	if not os.path.exists(flodername):
		os.mkdir(flodername)
	for ii_index in range(0, u):
		h_temp[ii_index, :] = np.squeeze(h_average[ii_index, int(s/2):s, :, :]).reshape((1, int(s/2)*cluster_num*cir_num), order='F')
		datatemp[0:np.size(datatemp, 0):2, :] = np.transpose(np.real(h_temp[ii_index, :][np.newaxis, :]))
		datatemp[1:np.size(datatemp, 0):2, :] = np.transpose(np.imag(h_temp[ii_index, :][np.newaxis, :]))
		datatemp = np.float32(datatemp)
		filename = '{1}/default_BS{2}_UE{3}_Ant{0}.bin'.format(ii_index, flodername, 0, 0)
		fw = open(filename, 'wb')
		fw.write(datatemp)
		fw.close
	return tmp







def	txtgenerate(floderName, channel_type, v, delay, h_average, upsampfactor):
	tmp = 'txt_FINISHED'
	if not os.path.exists(floderName):
		os.mkdir(floderName)
	filename = '{0}/{1}_{2}km_channel_vertex.txt'.format(floderName, channel_type.upper(), int(v))
	u = np.size(h_average, 0)
	s = np.size(h_average, 1)
	cluster_num = np.size(h_average, 2)
	cir_num = np.size(h_average, 3)
	h_average_real = np.real(h_average/np.max(np.abs(h_average)))
	h_average_imag = np.imag(h_average/np.max(np.abs(h_average)))
	#h_average_real = np.real(h_average)
	#h_average_imag = np.imag(h_average)
	delayw = ceil(delay * 1000)/10000

	configuration = ''
	for u_index in range(1, u+1):
		for s_index in range(1, s+1):
			configuration = configuration + 'A'+str(s_index)+'-'+'B'+str(u_index)+':'
			for nc_index in range(1, cluster_num+1):
				if nc_index == cluster_num:
					configuration = configuration + str(nc_index)+';'
				else:
					configuration = configuration + str(nc_index)+','

	for s_index in range(1, s+1):
		for u_index in range(1, u+1):
			configuration = configuration + 'B'+str(u_index)+'-'+'A'+str(s_index)+':'
			for nc_index in range(1, cluster_num+1):
				if nc_index == cluster_num and s_index != s and u_index != u:
					configuration = configuration + str(nc_index)+';'
				elif nc_index == cluster_num and s_index == s and u_index == u:
					configuration = configuration + str(nc_index)
				else:
					configuration = configuration + str(nc_index) + ','


	fw = open(filename, 'w')
	fw.write('[Spirent IQ Playback File]'+'\n')
	fw.write('Version = 1.0.0'+'\n')
	fw.write('Configuration = ' + configuration + '\n')
	fw.write('Upsample Factor = ' + str(upsampfactor)+'\n')
	fw.write('[Fading Sample Data]'+'\n')
	for cir_index in range(0, cir_num):
		for u_index in range(0, u):
			for s_index in range(0, s):
				for nc_index in range(0, cluster_num):
					tempdelay = '%.4f' % delayw[nc_index]
					ivalue = '%.4f' % h_average_real[u_index, s_index, nc_index, cir_index]
					qvalue = '%.4f' % h_average_imag[u_index, s_index, nc_index, cir_index]
					fw.write(str(tempdelay)+'\t'+str(ivalue)+'\t'+str(qvalue)+'\t')
		for s_index in range(0, s):
			for u_index in range(0, u):
				for nc_index in range(0, cluster_num):
					tempdelay = '%.4f' % delayw[nc_index]
					ivalue = '%.4f' % h_average_real[u_index, s_index, nc_index, cir_index]
					qvalue = '%.4f' % h_average_imag[u_index, s_index, nc_index, cir_index]
					if nc_index == cluster_num-1 and s_index == s-1 and u_index == u-1:
						fw.write(str(tempdelay)+'\t'+str(ivalue)+'\t'+str(qvalue))
					else:
						fw.write(str(tempdelay) + '\t' + str(ivalue) + '\t' + str(qvalue)+'\t')
		fw.write('\n')
	fw.close
	return tmp

