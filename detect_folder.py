import cv2
import os
from pathlib import Path

folder = 'burd3'
mark = False
x1 = 15
x2 = 525
y1 = 100
y2 = 350

bird_class = cv2.CascadeClassifier('cascade.xml')

# Opening image 
for im in os.listdir(folder):
    img = cv2.imread(f'{folder}/{im}')
    img_gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY) 

    # Use minSize because we're not 
    # bothering with extra-small 
    # dots that would look like birds
    minsize = 75
    found = bird_class.detectMultiScale(img_gray, minSize=(minsize, minsize)) 

    # Don't do anything if there's 
    # no bird
    amount_found = len(found) 
    print(f'Found {amount_found} birds in {im}')
    if amount_found > 0:
        
        # There may be more than one 
        # birdin the image 
        if mark:
            for (x, y, width, height) in found: 
                # We draw a green rectangle around 
                # every recognized bird 

                #adjust for roi
                x += x1
                y += y1

                cv2.rectangle(img, (x, y), 
                            (x + height, y + width), 
                            (0, 255, 0), 5) 
                
        path = Path(im)
        cv2.imwrite(f'out/{path.stem}_{amount_found}{path.suffix}', img)
        