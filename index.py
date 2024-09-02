import cv2
import numpy as np
from sklearn.cluster import KMeans

import tkinter as tk
from tkinter import Canvas, Frame
from PIL import Image, ImageTk, ImageFilter
import cv2
import numpy as np

texture_map = r'./texture_map.png'
labeled_texture_map = r'./labeled_texture_map.png'

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
    window_name = 'Reduced Color Image'
    cv2.imshow(window_name, result_img)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

class ImageApp:
    def __init__(self, root, original_img_path):
        self.root = root
        self.root.title("Image Threshold Adjuster")

        # Load original image
        self.original_img = cv2.imread(original_img_path)
        self.original_img_rgb = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
        self.original_pil = Image.fromarray(self.original_img_rgb)

        # Create canvas for displaying images
        self.canvas = Canvas(root, width=self.original_pil.width * 2, height=self.original_pil.height)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a frame for the scale and buttons
        self.control_frame = Frame(root)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Create Scale for threshold adjustment
        self.threshold_scale = tk.Scale(self.control_frame, from_=0, to_=255, orient=tk.VERTICAL, label="Threshold")
        self.threshold_scale.pack()

        # Bind the scale to update the image
        self.threshold_scale.bind("<Motion>", self.update_image)

        # Initial display
        self.update_image()

    def identify_islands(self, uv_img: cv2.typing.MatLike, threshold: int):
        # Load the texture map
        gray = cv2.cvtColor(uv_img, cv2.COLOR_BGR2GRAY)
        
        # Threshold the image to binary (black and white)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours which represent different islands
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create an empty image to draw contours
        output_img = np.zeros_like(uv_img)
        
        # Loop through the contours and draw them
        for i, contour in enumerate(contours):
            # Generate a color for each island (for visualization purposes)
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            
            # Draw the contour on the output image
            cv2.drawContours(output_img, [contour], -1, color, thickness=cv2.FILLED)
        return output_img

    def update_image(self, event=None):
        # Get the threshold value from the scale
        threshold = self.threshold_scale.get()

        thresh_img = self.identify_islands(self.original_img, threshold)

        # Convert to PIL format for display
        thresh_pil = Image.fromarray(thresh_img)

        # Convert images to PhotoImage format
        original_tk = ImageTk.PhotoImage(self.original_pil)
        thresh_tk = ImageTk.PhotoImage(thresh_pil)

        # Clear the canvas and display both images
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=original_tk)
        self.canvas.create_image(self.original_pil.width, 0, anchor=tk.NW, image=thresh_tk)

        # Update canvas size and scroll region
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

        # Keep references to avoid garbage collection
        self.original_tk = original_tk
        self.thresh_tk = thresh_tk


if __name__ == "__main__":
    # reduce_colors(image_path, k=1)
    # identify_islands(texture_map, labeled_texture_map)
    root = tk.Tk()
    app = ImageApp(root, texture_map)
    root.mainloop()
