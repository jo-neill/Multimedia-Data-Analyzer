import cv2
import numpy as np
import glob
from Tkinter import *
from collections import Counter

def callback():
    img_name = E1.get()

####################
    
    #Initialization variables
    hsv_imgs = []
    hsv_means = []
    img_hists = []
    img_labels = []
    img_size = 250;
    img_int_size = (img_size**2) * 3

    #Get user's image
    input_img = img_name

    print('Fetching..')
    img = cv2.imread(input_img)
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
    img_labels.append(-1)

    ##############################
    #Prepare images for processing
    ##############################

    SZ=20
    bin_n = 16 # Number of bins

    svm_params = dict( kernel_type = cv2.ml.SVM_LINEAR, svm_type = cv2.ml.SVM_C_SVC, C=2.67, gamma=5.383 )

    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

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

            hsv_imgs.append(img_hsv)
            img_hists.append(hist)
            img_labels.append(5)

            continue
        else:
            continue

    #Adds unfiltered images as HSV to training data
    print('Formatting Unfiltered Images')
    for filename in glob.glob('unfiltered/*'):
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


            hsv_imgs.append(img_hsv)
            img_hists.append(hist)
            img_labels.append(6)

            continue
        else:
            continue

    hsv_means = np.float32(hsv_means)

    #print('Separating Data into Training and Testing')
    test_hog_descriptors = []
    train_hog_descriptors = []
    train_means = []
    test_means = []
    test_lbls = []
    train_lbls = []
    slice_interval = 100
    set_quant = 100
    sets = 6
    test_num = 0


    for x in range(0, sets):
        for k in range(0, set_quant):
            if x == 0 and k == 0:
                test_lbls.append(img_labels[x*set_quant+k])
                test_means.append(hsv_means[x*set_quant+k])
                test_num += 1
            else:
                train_lbls.append(img_labels[x*set_quant+k])
                train_means.append(hsv_means[x*set_quant+k])         
            continue

    train_means = np.array(train_means)
    test_means = np.array(test_means)
    test_hog_descriptors = np.squeeze(test_hog_descriptors)
    train_hog_descriptors = np.squeeze(train_hog_descriptors)
    test_lbls = np.array(test_lbls)
    train_lbls = np.array(train_lbls)

    print('Training & Testing KNN Classifier')
    # train the k-Nearest Neighbor classifier with the current value of `k`
    # Initiate kNN, train the data, then test it with test data for k=1
    results = np.array((0))

    for k in range (1, 30, 2):
        knn = cv2.ml.KNearest_create()
        knn.train(train_means, cv2.ml.ROW_SAMPLE, train_lbls)
        ret,result,neighbours,dist = knn.findNearest(test_means, k)

        # Now we check the accuracy of classification
        # For that, compare the result with test_labels and check which are wrong
        filter_guess = ""
        results = np.append(results, result[0])
        continue

    data = Counter(results)
    print(data.most_common())

    guess = data.most_common(1)[0][0]

    if guess == 1:
        filter_guess = ('Your image may have the "Warm" filter applied.')
    if guess == 2:
        filter_guess = ('Your image may have "Superfade" filter applied.')
    if guess == 3:
        filter_guess = ('Your image may have "Noir" filter applied.')
    if guess == 4:
        filter_guess = ('Your image may have "Antique" filter applied.')
    if guess == 5:
        filter_guess = ('Your image may have "Bleached" filter applied.')
    if guess == 6:
        filter_guess = ('Your image may not have a filter.')



    L2 = Label(top, text= filter_guess, font=("Courier New", 14))
    L2.place(relx=.5, rely=.75, anchor=CENTER)

top = Tk()
top.title("Filter Identifier")
top.minsize(width=300, height=300)
L1 = Label(top, text="Image Name:", font=("Courier New", 20, "bold"))
L1.grid(row=5, column=0)
E1 = Entry(top, bd = 5, font=("Courier New", 20))
E1.grid(row=5, column=1)

MyButton1 = Button(top, text="Submit", width=10, font=("Courier New", 20), command=callback)
MyButton1.place(relx=.5, rely=.5, anchor=CENTER)
#MyButton1.grid(row=15, column=1)

top.mainloop()