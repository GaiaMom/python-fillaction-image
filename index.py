import cv2
import numpy as np
from sklearn.cluster import KMeans

def reduce_colors(image_path, k=1):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to be a list of pixels
    pixels = image_rgb.reshape(-1, 3)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    
    # Get the color of the cluster center
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    
    # Replace all pixels with the dominant color
    segmented_img = np.full_like(image_rgb, dominant_color)
    
    # Convert the result back to BGR for display/saving with OpenCV
    result_img = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR)
    
    # Save or display the result
    cv2.imwrite('result_image.png', result_img)
    cv2.imshow('Reduced Color Image', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to your image
image_path = r'./texture_map.png'
reduce_colors(image_path, k=1)
