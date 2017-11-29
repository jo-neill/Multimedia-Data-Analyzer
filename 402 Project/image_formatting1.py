import cv2
import numpy as np
import glob

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  # Known bug: https://github.com/opencv/opencv/issues/4969
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 12.5, gamma = 0.50625):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):

        return self.model.predict(samples)[1].ravel()

def evaluate_model(model, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print('Accuracy: %.2f %%' % ((1 - err)*100))

    confusion = np.zeros((10, 10), np.int32)
    for i, j in zip(labels, resp):
        confusion[int(i), int(j)] += 1
    print('confusion matrix:')
    print(confusion)

def get_hog() : 
    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)

    return hog
    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR





hsv_imgs = []
img_hists = []
img_labels = []
img_size = 250;
img_int_size = (img_size**2) * 3

#Adds warm filter images as HSV to training data
print('Formatting Warm Images')
for filename in glob.glob('warm/*'):
    if filename.endswith(".jpg") or filename.endswith(".png"): 
        img = cv2.imread(filename)
        height, width = img.shape[:2]
        max_height = img_size
        max_width = img_size

        # only shrink if img is bigger than required
        #if max_height < height or max_width < width:
            # get scaling factor
        scaling_factorY = max_height / float(height)
            #if max_width/float(width) < scaling_factor:
        scaling_factorX = max_width / float(width)
            # resize image
        img = cv2.resize(img, None, fx=scaling_factorX, fy=scaling_factorY, interpolation=cv2.INTER_AREA)

        #key = cv2.waitKey()

	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([img_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

	hsv_imgs.append(img_hsv)
	img_hists.append(hist)
	img_labels.append(0)

        continue
    else:
        continue

print(len(hsv_imgs))
print(len(img_hists))
print(len(img_labels))

#Adds superfade filter images as HSV to training data
print('Formatting Superfade Images')
for filename in glob.glob('superfade/*'):
    if filename.endswith(".jpg") or filename.endswith(".png"): 
        img = cv2.imread(filename)
        height, width = img.shape[:2]
        max_height = img_size
        max_width = img_size

        # only shrink if img is bigger than required
        #if max_height < height or max_width < width:
            # get scaling factor
        scaling_factorY = max_height / float(height)
            #if max_width/float(width) < scaling_factor:
        scaling_factorX = max_width / float(width)
            # resize image
        img = cv2.resize(img, None, fx=scaling_factorX, fy=scaling_factorY, interpolation=cv2.INTER_AREA)

        #key = cv2.waitKey()

	hist = cv2.calcHist([img_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

	hsv_imgs.append(img_hsv)
	img_hists.append(hist)
	img_labels.append(1)

        continue
    else:
        continue

#Adds noir filter images as HSV to training data
print('Formatting Noir Images')
for filename in glob.glob('noir/*'):
    if filename.endswith(".jpg") or filename.endswith(".png"): 
        img = cv2.imread(filename)
        height, width = img.shape[:2]
        max_height = img_size
        max_width = img_size

        # only shrink if img is bigger than required
        #if max_height < height or max_width < width:
            # get scaling factor
        scaling_factorY = max_height / float(height)
            #if max_width/float(width) < scaling_factor:
        scaling_factorX = max_width / float(width)
            # resize image
        img = cv2.resize(img, None, fx=scaling_factorX, fy=scaling_factorY, interpolation=cv2.INTER_AREA)

        #key = cv2.waitKey()

	hist = cv2.calcHist([img_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

	hsv_imgs.append(img_hsv)
	img_hists.append(hist)
	img_labels.append(2)

        continue
    else:
        continue

test_imgs = []
train_imgs = []
test_lbls = []
train_lbls = []
temp_image = []

print('Defining HoG parameters ...')
# HoG feature descriptor
hog = get_hog();

print('Calculating HoG descriptor for every image ... ')
hsv_imgs = np.array(hsv_imgs)
hog_descriptors = []
for img in hsv_imgs:
    hog_descriptors.append(hog.compute(img))
hog_descriptors = np.squeeze(hog_descriptors)

print('Training SVM model ...')
img_labels = np.array(img_labels)
model = SVM()
model.train(hog_descriptors, img_labels)

print('Saving SVM model ...')
model.save('filters_svm.dat')

print('Evaluating model ... ')
vis = evaluate_model(model, hog_descriptors, img_labels)
cv2.waitKey(0)











#print (len(test_images))
#print (hsv_imgs)
#print (len(hsv_imgs))
#print (len(img_labels))





#im = cv2.imread("unfiltered.jpg")
#im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

#cv2.imshow("hsv image", im_hsv)

#blue = numpy.uint8([[[0,0,255]]])
#hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
#print(hsv_blue)