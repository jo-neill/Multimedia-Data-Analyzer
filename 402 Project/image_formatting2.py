import cv2
import numpy as np
import glob

SZ=20
bin_n = 16 # Number of bins

svm_params = dict( kernel_type = cv2.ml.SVM_LINEAR, svm_type = cv2.ml.SVM_C_SVC, C=2.67, gamma=5.383 )

affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

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

    confusion = np.zeros((5, 5), np.int32)
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
hsv_means = []
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

        hue, sat, val = img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2]

        hue_mean = np.mean(hue)
        sat_mean = np.mean(sat)
        val_mean = np.mean(val)

        hue_range = np.ptp(hue)
        sat_range = np.ptp(sat)
        val_range = np.ptp(val)

        hsv_means.append([hue_mean, sat_mean, val_mean, hue_range, sat_range, val_range])


        hist = cv2.calcHist(img_hsv, [0, 1], None, [180, 256], [0, 180, 0, 256])

        hsv_imgs.append(img_hsv)
        img_hists.append(hist)
        img_labels.append(1)

        continue
    else:
        continue

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

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hue, sat, val = img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2]

        hue_mean = np.mean(hue)
        sat_mean = np.mean(sat)
        val_mean = np.mean(val)

        hue_range = np.ptp(hue)
        sat_range = np.ptp(sat)
        val_range = np.ptp(val)

        hsv_means.append([hue_mean, sat_mean, val_mean, hue_range, sat_range, val_range])

    	hist = cv2.calcHist([img_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    	hsv_imgs.append(img_hsv)
    	img_hists.append(hist)
    	img_labels.append(2)

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

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hue, sat, val = img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2]

        hue_mean = np.mean(hue)
        sat_mean = np.mean(sat)
        val_mean = np.mean(val)

        hue_range = np.ptp(hue)
        sat_range = np.ptp(sat)
        val_range = np.ptp(val)

        hsv_means.append([hue_mean, sat_mean, val_mean, hue_range, sat_range, val_range])

    	hist = cv2.calcHist([img_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    	hsv_imgs.append(img_hsv)
    	img_hists.append(hist)
    	img_labels.append(3)

        continue
    else:
        continue

#Adds antique filter images as HSV to training data
print('Formatting Antique Images')
for filename in glob.glob('antique/*'):
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

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hue, sat, val = img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2]

        hue_mean = np.mean(hue)
        sat_mean = np.mean(sat)
        val_mean = np.mean(val)

        hue_range = np.ptp(hue)
        sat_range = np.ptp(sat)
        val_range = np.ptp(val)

        hsv_means.append([hue_mean, sat_mean, val_mean, hue_range, sat_range, val_range])

        hist = cv2.calcHist([img_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

        hsv_imgs.append(img_hsv)
        img_hists.append(hist)
        img_labels.append(4)

        continue
    else:
        continue

#Adds bleached filter images as HSV to training data
print('Formatting Bleached Images')
for filename in glob.glob('bleached/*'):
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

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hue, sat, val = img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2]

        hue_mean = np.mean(hue)
        sat_mean = np.mean(sat)
        val_mean = np.mean(val)

        hue_range = np.ptp(hue)
        sat_range = np.ptp(sat)
        val_range = np.ptp(val)

        hsv_means.append([hue_mean, sat_mean, val_mean, hue_range, sat_range, val_range])

        hist = cv2.calcHist([img_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

        hsv_imgs.append(img_hsv)
        img_hists.append(hist)
        img_labels.append(5)

        continue
    else:
        continue

# #Adds unfiltered images as HSV to training data
# print('Formatting Unfiltered Images')
# for filename in glob.glob('unfiltered/*'):
#     if filename.endswith(".jpg") or filename.endswith(".png"): 
#         img = cv2.imread(filename)
#         height, width = img.shape[:2]
#         max_height = img_size
#         max_width = img_size

#         # only shrink if img is bigger than required
#         #if max_height < height or max_width < width:
#             # get scaling factor
#         scaling_factorY = max_height / float(height)
#             #if max_width/float(width) < scaling_factor:
#         scaling_factorX = max_width / float(width)
#             # resize image
#         img = cv2.resize(img, None, fx=scaling_factorX, fy=scaling_factorY, interpolation=cv2.INTER_AREA)

#         img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#         hue, sat, val = img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2]

#         hue_mean = np.mean(hue)
#         sat_mean = np.mean(sat)
#         val_mean = np.mean(val)

#         hue_range = np.ptp(hue)
#         sat_range = np.ptp(sat)
#         val_range = np.ptp(val)

#         hsv_means.append([hue_mean, sat_mean, val_mean, hue_range, sat_range, val_range])

#         hist = cv2.calcHist([img_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

#         hsv_imgs.append(img_hsv)
#         img_hists.append(hist)
#         img_labels.append(6)

#         continue
#     else:
#         continue

# print('Defining HoG parameters ...')
# # HoG feature descriptor
# hog = get_hog();

# print('Calculating HoG descriptor for every image ... ')
# hsv_imgs = np.array(hsv_imgs)
# hog_descriptors = []
# for img in hsv_imgs:
#     hog_descriptors.append(hog.compute(img))
# hog_descriptors = np.squeeze(hog_descriptors)

# print('Shuffle data ... ')
# # Shuffle data
# rand = np.random.RandomState(10)
# shuffle = rand.permutation(len(hog_descriptors))
# hog_descriptors, img_labels = hog_descriptors[shuffle], img_labels[shuffle]

hsv_means = np.float32(hsv_means)
master_acc = 0
master_min_acc = 100
master_k = 0
master_accs = []
best_z = 0

for z in range(0, 3000):
    #print('Separating Data into Training and Testing')
    test_hog_descriptors = []
    train_hog_descriptors = []
    train_means = []
    test_means = []
    test_lbls = []
    train_lbls = []
    slice_interval = 75
    set_quant = 100
    sets = 3
    test_num = 0


    for x in range(0, sets):
        for k in range(0, set_quant):
            sorter = np.random.randint(0, set_quant)
            if (sorter < slice_interval):
                temp = img_hists[x*set_quant+k]
                temp = np.mean(temp)
                train_hog_descriptors.append(temp)
                train_lbls.append(img_labels[x*set_quant+k])
                train_means.append(hsv_means[x*set_quant+k])
            else:
                temp = img_hists[x*set_quant+k]
                temp = np.mean(temp)
                test_hog_descriptors.append(temp)
                test_lbls.append(img_labels[x*set_quant+k])
                test_means.append(hsv_means[x*set_quant+k])
                test_num += 1
            continue

    # print(len(train_images))
    # print(train_images)
    # print(len(test_images))
    # print(len(train_lbls))
    # print(len(test_lbls))

    train_means = np.array(train_means)
    test_means = np.array(test_means)
    test_hog_descriptors = np.squeeze(test_hog_descriptors)
    train_hog_descriptors = np.squeeze(train_hog_descriptors)
    test_lbls = np.array(test_lbls)
    train_lbls = np.array(train_lbls)

    best_k = 1
    best_accuracy = 0

    #print('Training & Testing KNN Classifier')
    for k in range(1, 30, 2):
        # train the k-Nearest Neighbor classifier with the current value of `k`
        # Initiate kNN, train the data, then test it with test data for k=1
        knn = cv2.ml.KNearest_create()
        knn.train(train_means, cv2.ml.ROW_SAMPLE, train_lbls)
        ret,result,neighbours,dist = knn.findNearest(test_means, k)

        # Now we check the accuracy of classification
        # For that, compare the result with test_labels and check which are wrong

        x = 0
        full_results = []
        correct = 0.0

        while x < len(test_lbls) -1:
            full_results.append([result[x], test_lbls[x]])
            x += 1
            if result[x] == test_lbls[x]:
                correct += 1
        accuracy = correct/test_num * 100

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
        continue

    master_accs.append(best_accuracy)

    if best_accuracy < master_min_acc:
        master_min_acc = best_accuracy

    if best_accuracy > master_acc:
        master_acc = best_accuracy
        master_k = best_k
        best_z = z

    if z % 100 == 0:
        print(z/30)
        print('Current Best Accuracy:')
        print(master_acc)
    # print('Accuracy:')
    # print(best_accuracy)
    # print('Best K Value:')
    # print(best_k)

print('Overall Best:')
print(master_acc)
print(master_k)
print('Average Accuracy:')
print(np.mean(master_accs))
print('Overall Worst:')
print(master_min_acc)
print('Spread:')
print(master_acc-master_min_acc)
print('Found on:')
print(best_z)
 









# ######     Now training      ########################

# svm = cv2.ml.SVM_create()
# svm.train(train_hog_descriptors, cv2.ml.ROW_SAMPLE, train_lbls)
# svm.save('svm_data.dat')

# ######     Now testing      ########################

# result = svm.predict(test_hog_descriptors)
# print(result)

# #######   Check Accuracy   ########################
# mask = result==test_lbls
# correct = np.count_nonzero(mask)
# print correct*100.0/len(result)





# print('Training SVM model ...')
# model = SVM()
# model.train(train_hog_descriptors, train_lbls)



# print('Saving SVM model ...')
# model.save('filters_svm.dat')

# print('Evaluating model ... ')
# vis = evaluate_model(model, test_hog_descriptors, test_lbls)
# cv2.waitKey(0)
