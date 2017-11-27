import flickrapi
import urllib
import os
import config
from random import randint
import time

api_key = '49dde3b8b8ff468446fd2d773666b7ad'
api_secret = 'aad2a1f30b38e273'
img_folder = "Users/Joe/Downloads/FlickrImages"

download_flickr_photos('banana', size='square',max_nb_img=5)

def download_flickr_photos(keywords, size='original', max_nb_img=-1):
    """
    Downloads images based on keyword search on the Flickr website
    
    Parameters
    ----------
    keywords : string, list of strings
        Keyword to search for or a list of keywords should be given.
    size : one of the following strings 'thumbnail', 'square', 'medium', default: 'original'.
        Size of the image to download. In this function we only provide
        four options. More options are explained at 
        http://librdf.org/flickcurl/api/flickcurl-searching-search-extras.html
    max_nb_img : int, default: -1
        Maximum number of images per keyword to download. If given a value of -1, all images
        will be downloaded
    
    Returns
    ------
    Images found based on the keyword are saved in a separate subfolder.
    
    Notes
    -----
    This function uses the Python package flickrapi and its walk method. 
    FlickrAPI.walk has same parameters as FlickrAPI.search
    http://www.flickr.com/services/api/flickr.photos.search.html
    
    To use the Flickr API a set of API keys needs to be created on 
    https://www.flickr.com/services/api/misc.api_keys.html
    """
    if not (isinstance(keywords, str) or isinstance(keywords, list)):
        raise AttributeError('keywords must be a string or a list of strings')
        
    if not (size in ['thumbnail', 'square', 'medium', 'original']):
        raise AttributeError('size must be "thumbnail", "square", "medium" or "original"')
                             
    if not (max_nb_img == -1 or (max_nb_img > 0 and isinstance(max_nb_img, int))):
        raise AttributeError('max_nb_img must be an integer greater than zero or equal to -1')
    
    flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
    
    if isinstance(keywords, str):
        keywords_list = []
        keywords_list.append(keywords)
    else:
        keywords_list = keywords
        
    if size == 'thumbnail':
        size_url = 'url_t'
    elif size == 'square':
        size_url = 'url_q'
    elif size == 'medium':
        size_url = 'url_c'
    elif size == 'original':
        size_url = 'url_o'
    
    for keyword in keywords_list:
        count = 0
                             
        #print('Downloading images for', keyword)

        results_folder = img_folder + keyword.replace(" ", "_") + "/"
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        photos = flickr.walk(
                     text=keyword,
                     extras=size_url,
                     license='1,2,4,5',
                     per_page=50)
        
        urls = []
        for photo in photos:
            t = randint(1, 3)
            time.sleep(t)
            count += 1
            if max_nb_img != -1:
                if count > max_nb_img:
                    print('Reached maximum number of images to download')
                    break
            try:
                url=photo.get(size_url)
                urls.append(url)
                
                urllib.request.urlretrieve(url,  results_folder + str(count) +".jpg")
                print('Downloading image #' + str(count) + ' from url ' + url)
            except Exception as e:
                print(e, 'Download failure')
                             
        print("Total images downloaded:", str(count - 1))