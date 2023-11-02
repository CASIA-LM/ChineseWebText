"""
Author: Subhabrata Mukherjee (submukhe@microsoft.com)
Code for Uncertainty-aware Self-training (UST) for few-shot learning.
"""

from sklearn.utils import shuffle

import logging
import numpy as np

logger = logging.getLogger('UST')

def get_BALD_acquisition(y_T):
	num_sample, T = y_T.shape
	y_T_ = np.zeros((num_sample, T, 2))
	y_T_[:, :, 0] = y_T
	y_T_[:, :, 1] = 1- y_T
	expected_entropy = - np.mean(np.sum(y_T_ * np.log(y_T_ + 1e-10), axis=-1), axis=1)
	expected_p = np.mean(y_T_, axis=1)
	entropy_expected_p = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)

	return (entropy_expected_p - expected_entropy)


def sample_by_bald_difficulty(ids, y_var, y, num_samples, label_type, y_T):

	logger.info ("Sampling by difficulty BALD acquisition function")
	BALD_acq = get_BALD_acquisition(y_T)
	p_norm = np.maximum(np.zeros(len(BALD_acq)), BALD_acq)
	p_norm = p_norm / np.sum(p_norm)
	indices = np.random.choice(len(ids), num_samples, p=p_norm, replace=False)
	y_s = y[indices]
	w_s = y_var[indices][:, 0]
	return ids[indices], y_s, w_s


def sample_by_bald_easiness(ids, y_var, y, num_samples, label_type, y_T):

	logger.info ("Sampling by easy BALD acquisition function")
	BALD_acq = get_BALD_acquisition(y_T) # y_T is of shape (T, num_samples, num_classes)
	p_norm = np.maximum(np.zeros(len(BALD_acq)), (1. - BALD_acq)/np.sum(1. - BALD_acq))
	p_norm = p_norm / np.sum(p_norm)
	logger.info (p_norm[:10])
	indices = np.random.choice(len(ids), num_samples, p=p_norm, replace=False)
	y_s = y[indices]
	w_s = y_var[indices][:,0]
	return ids[indices], y_s, w_s



def sample_by_bald_class_easiness(ids, y_var, y, num_samples, label_type, y_T):

	logger.info ("Sampling by easy BALD acquisition function per class")
	BALD_acq = get_BALD_acquisition(y_T)
	BALD_acq = (BALD_acq - np.min(BALD_acq)) / (np.max(BALD_acq) - np.min(BALD_acq))
	BALD_acq = (1. - BALD_acq)/np.sum(1. - BALD_acq)
	logger.info (BALD_acq)
	samples_per_class = num_samples // len(label_type)
	ids_list, y_s, w_s = [], [], []
	for label in label_type:
		y_ = y[y==label]
		y_var_ = y_var[y == label]	
		ids_ = ids[y == label]		
		# p = y_mean[y == label]
		p_norm = BALD_acq[y==label]
		p_norm = np.maximum(np.zeros(len(p_norm)), p_norm)
		p_norm = p_norm/np.sum(p_norm)
		if len(y_) < samples_per_class:
			logger.info ("Sampling with replacement.")
			replace = True
		else:
			replace = False
		indices = np.random.choice(len(y_), samples_per_class, p=p_norm, replace=replace)
		
		ids_list.extend(ids_[indices].tolist())
		y_s.extend(y_[indices].tolist())
		w_s.extend(y_var_[indices].tolist())

	return np.array(ids_list), np.array(y_s), np.array(w_s)


def sample_by_bald_class_difficulty(ids, y_var, y, num_samples, label_type, y_T):

	logger.info ("Sampling by difficulty BALD acquisition function per class")
	BALD_acq = get_BALD_acquisition(y_T)
	samples_per_class = num_samples // len(label_type)
	ids_list, y_s, w_s = [], [], []

	for label in label_type:
		y_ = y[y==label]
		y_var_ = y_var[y == label]		
		p_norm = BALD_acq[y==label]
		p_norm = np.maximum(np.zeros(len(p_norm)), p_norm)
		p_norm = p_norm/np.sum(p_norm)
		if len(y_) < samples_per_class:
			samples_per_class == len(y_)
		
		indices = np.random.choice(len(y_), samples_per_class, p=p_norm)
		ids_list.extend(ids[indices].tolist())
		y_s.extend(y_[indices].tolist())
		w_s.extend(y_var_[indices].tolist())

	return np.array(ids_list), np.array(y_s), np.array(w_s)


def uniform_sample(ids, y, y_var, num_samples, label_type):
	ids_list, y_list, w_s = [], [], []
	samples_per_class = num_samples // len(label_type)
	for label in label_type:
		ids_ = ids[y == label]
		y_ = y[y == label]
		y_var_ = y_var_[y == label]
		if len(ids_) < samples_per_class:
			replace = True
			logger.info ("Sampling with replacement.")
		else:
			replace = False
		indices = np.random.choice(len(ids_), samples_per_class, replace=replace)
		ids_list.extend(ids_[indices].tolist())
		y_list.extend(y_[indices].tolist())
	return np.array(ids_list), np.array(y_list)


def sample_by_score_class_easy(ids, y, y_mean, y_var, num_samples):
	logger.info ("Sampling by difficulty BALD acquisition function per class")
	samples_per_class = num_samples // 2
	ids_list, y_s, w_s = [], [], []

	for label in (0, 1):
		y_ = y[y==label]
		y_var_ = y_var[y == label]	
		y_mean_ = y_mean[y == label]		
		ids_ = ids[y == label]	

		if label == 0:
			misc = (1 - y_mean_) / (np.sum(1 - y_mean_))
		else:
			misc = y_mean_ / np.sum(y_mean_)

		p_norm = misc
		p_norm = np.maximum(np.zeros(len(p_norm)), p_norm)
		p_norm = p_norm/np.sum(p_norm)

		if len(y_) < samples_per_class:
			replace = True
			logger.info ("Sampling with replacement.")
		else:
			replace = False

		indices = np.random.choice(len(y_), samples_per_class, p=p_norm, replace=replace)
		ids_list.extend(ids_[indices].tolist())
		y_s.extend(y_[indices].tolist())
		w_s.extend(y_var_[indices].tolist())

	return np.array(ids_list), np.array(y_s), np.array(w_s)