import insightface
import numpy as np
from numpy.linalg import norm

def verification(img1, img2, model):
	sim2 = model.compute_sim(img1, img2)
	sim = 0.5 * (sim2+1)
	return sim