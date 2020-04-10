import cv2
import os

bird_class = cv2.CascadeClassifier('cascade.xml')

# Opening image 
for im in os.listdir('bird_images'):
    img = cv2.imread(f'bird_images/{im}')

    x1 = 0
    x2 = img.shape[1]
    y1 = 100
    y2 = 350
    img_gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY) 
    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    # Use minSize because for not 
    # bothering with extra-small 
    # dots that would look like birds
    found = bird_class.detectMultiScale(img_gray, 
                                    minSize =(100, 100)) 

    # Don't do anything if there's 
    # no bird
    amount_found = len(found) 
    print(f'Found {amount_found} birds in {im}')
    if amount_found > 0:
        
        # There may be more than one 
        # birdin the image 
        for (x, y, width, height) in found: 
            # We draw a green rectangle around 
            # every recognized bird 

            #adjust for roi
            x += x1
            y += y1

            cv2.rectangle(img, (x, y), 
                        (x + height, y + width), 
                        (0, 255, 0), 5) 
            
        # Creates the environment of 
        # the picture and shows it 
        #plt.subplot(1, 1, 1) 
        #plt.imshow(img_rgb) 
        #plt.show()
        #with open(, 'wb') as f:
            #plt.savefig(f)
        cv2.imwrite(f'out/{amount_found}_{im}', img)
        