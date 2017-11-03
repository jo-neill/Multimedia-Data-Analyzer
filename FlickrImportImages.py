import flickrapi

api_key = u'49dde3b8b8ff468446fd2d773666b7ad'
api_secret = u'aad2a1f30b38e273'

flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
#api reference = https://www.flickr.com/services/api/flickr.photosets.getPhotos.html
album = flickr.photosets.getPhotos(photoset_id='72157687504236930', user_id = '159463028@N08', extras = 'tags, machine_tags')
#title  = album['photosets']['photoset'][0]['title']['_content']

print (album['photoset']['photo'][0]['tags']['_content'])