import cv2
import numpy as np
from sklearn.cluster import KMeans

import tkinter as tk
from tkinter import Canvas, Frame
from PIL import Image, ImageTk, ImageFilter
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import math

texture_map = r'./labeled_texture_map_1.png'
labeled_texture_map = r'./labeled_texture_map_2.png'

class ContourImg:
    def identify_islands(uv_img: cv2.typing.MatLike, threshold: int):
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

    def fill_contours(uv_img, threshold, min_contour_area=100):
        # Convert to grayscale
        gray_img = cv2.cvtColor(uv_img, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred_img, threshold, 255)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty image to draw filled contours
        filled_contours_img = np.zeros_like(uv_img)

        # Iterate through contours and filter by area
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_contour_area:
                cv2.drawContours(filled_contours_img, [contour], -1, (0, 255, 0), thickness=cv2.FILLED)
        return filled_contours_img

    def detect_edges_rgb(uv_img: cv2.typing.MatLike, threshold):
        uv_img = cv2.GaussianBlur(uv_img, (5, 5), 0)
        # Split the image into its B, G, R channels
        b_channel, g_channel, r_channel = cv2.split(uv_img)

        # Apply Canny edge detection on each channel
        edges_b = cv2.Canny(b_channel, threshold, 255)
        edges_g = cv2.Canny(g_channel, threshold, 255)
        edges_r = cv2.Canny(r_channel, threshold, 255)

        # Combine the edges from each channel
        edges_combined = np.maximum(edges_b, np.maximum(edges_g, edges_r))

        # Convert edges to a 3-channel image
        edges_colored = cv2.cvtColor(edges_combined, cv2.COLOR_GRAY2BGR)

        # Overlay edges on the original RGB image
        overlay_img = cv2.addWeighted(uv_img, 0.8, edges_colored, 1.0, 0)

        return overlay_img

    def adjust_saturation_value(uv_img, s_threshold, v_threshold):
        
        # Convert the image from RGB to HSV
        hsv_image = cv2.cvtColor(uv_img, cv2.COLOR_BGR2HSV)
        
        # Extract HSV channels
        h, s, v = cv2.split(hsv_image)
        
        # Set saturation and value to 100%
        s[:] = s_threshold  # Saturation max value in OpenCV
        v[:] = v_threshold  # Value max value in OpenCV
        
        # Merge the channels back
        hsv_image = cv2.merge([h, s, v])
        
        # Convert the HSV image back to RGB
        rgb_adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        
        return rgb_adjusted_image
    
class ClusteringImgColor:    
    def convert_rgb_to_hsv(image):
        # Convert from RGB to HSV using OpenCV
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    def cluster_pixels(image, n_clusters=5):
        # Reshape the image to a 2D array of pixels
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # Create and fit the KMeans model
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)
        
        # Get the cluster centers (mean colors)
        cluster_centers = kmeans.cluster_centers_.astype(np.uint8)
        
        # Predict the cluster for each pixel
        labels = kmeans.predict(pixels)
        
        return labels, cluster_centers

    def apply_kmeans_to_image(labels, cluster_centers, image_shape):
        # Map each pixel to its cluster center color
        new_pixels = np.array([cluster_centers[label] for label in labels], dtype=np.uint8)
        
        # Reshape the result back to the original image shape
        new_image = new_pixels.reshape(image_shape)
        
        return new_image

    def convert_hsv_to_rgb(image):
        # Convert from HSV to RGB using OpenCV
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    def kmean_clustering(image, n_clusters):
        # Convert RGB to HSV
        hsv_image = ClusteringImgColor.convert_rgb_to_hsv(image)
        
        # Cluster the pixels in the HSV color space
        labels, cluster_centers = ClusteringImgColor.cluster_pixels(hsv_image, n_clusters=n_clusters)  # Adjust number of clusters as needed
        
        # Apply the K-Means result to the image
        clustered_hsv_image = ClusteringImgColor.apply_kmeans_to_image(labels, cluster_centers, hsv_image.shape)
        
        # Convert the clustered HSV image back to RGB
        clustered_rgb_image = ClusteringImgColor.convert_hsv_to_rgb(clustered_hsv_image)
        
        # Convert RGB back to BGR for OpenCV
        clustered_bgr_image = cv2.cvtColor(clustered_rgb_image, cv2.COLOR_RGB2BGR)

        return clustered_rgb_image, clustered_bgr_image

class GetMainColorMap:
    # Load and preprocess the image
    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb

    def preprocess_image(self, image_rgb):
        pixels = image_rgb.reshape(-1, 3)
        return pixels

    # Apply K-means clustering
    def apply_kmeans(self, pixels, num_colors):
        kmeans = KMeans(n_clusters=num_colors, random_state=0)
        kmeans.fit(pixels)
        clusters = kmeans.cluster_centers_
        labels = kmeans.labels_
        color_counts = dict(zip(*np.unique(labels, return_counts=True)))
        return clusters, color_counts
    
    def find_first_index_less_than(self, relative_changes, threshold=0.1):
        # Find the first index where the value is less than the threshold
        for index, value in enumerate(relative_changes):
            if value < threshold:
                return index
        return -1  # Return -1 if no value is less than the threshold
    
    # Plot the main colors
    def plot_colors(self, colors, counts):
        plt.figure(figsize=(8, 6))
        plt.title(f"Main Colors")
        plt.bar(range(len(colors)), counts, color=np.array(colors)/255)
        plt.xlabel('Color')
        plt.ylabel('Count')
        plt.xticks(range(len(colors)), [f'Color {i+1}' for i in range(len(colors))], rotation=90)
        plt.show()

    # Find the optimal number of clusters using the Elbow Method
    def find_optimal_k(self, pixels, max_k=10):
        wcss = []
        for k in range(1, max_k+1):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)
            wcss.append(kmeans.inertia_)

        print("Relative Changes (%):", self.calculate_relative_change(wcss))

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, max_k+1), wcss, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()

        return np.array(wcss), self.calculate_relative_change(wcss)
    
    def calculate_relative_change(self, array):
        if len(array) < 2:
            return np.array([])  # Not enough data to compute relative change
        
        array = np.array(array)
        
        # Handle zero values to avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_change = abs(array[1:] - array[:-1]) / array.max() * 100
        
        return relative_change

    # Main function to execute the script
    def detect(self, image_path, num_colors=5):
        image_rgb = self.load_image(image_path)
        pixels = self.preprocess_image(image_rgb)

        # Find the optimal number of clusters (optional)
        wcss, relative_chg = self.find_optimal_k(pixels, max_k=10)

        if (self.find_first_index_less_than(relative_chg) != -1):
            num_colors = self.find_first_index_less_than(relative_chg) + 1

        # Apply K-means clustering with the chosen number of colors
        clusters, color_counts = self.apply_kmeans(pixels, num_colors)
        
        # Prepare and plot the results
        main_colors = np.round(clusters).astype(int)
        color_counts_list = [color_counts.get(i, 0) for i in range(num_colors)]
        self.plot_colors(main_colors, color_counts_list)

class ImageAppUsingML:
    def __init__(self, root, original_img_path, n_cluster = 10):
        self.n_cluster = n_cluster
        
        self.root = root
        self.root.title("Image App Using ML")

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
        self.threshold_scale = tk.Scale(self.control_frame, from_=10, to_=20, orient=tk.VERTICAL, label="Threshold")
        self.threshold_scale.pack()
        self.threshold_scale_1 = tk.Scale(self.control_frame, from_=0, to_=255, orient=tk.VERTICAL, label="Threshold_1")
        self.threshold_scale_1.pack()

        # Bind the scale to update the image
        self.threshold_scale.bind("<Motion>", self.update_image)
        self.threshold_scale_1.bind("<Motion>", self.update_image)

        # Initial display
        self.update_image()

    def update_image(self, event=None):
        # Get the threshold value from the scale
        threshold = self.threshold_scale.get()
        threshold_1 = self.threshold_scale_1.get()

        clustered_rgb_image, clustered_bgr_image = ClusteringImgColor.kmean_clustering(self.original_img_rgb, self.n_cluster)

        thresh_img = clustered_rgb_image
        cv2.imwrite(labeled_texture_map, clustered_bgr_image)

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
        self.threshold_scale_1 = tk.Scale(self.control_frame, from_=0, to_=255, orient=tk.VERTICAL, label="Threshold_1")
        self.threshold_scale_1.pack()

        # Bind the scale to update the image
        self.threshold_scale.bind("<Motion>", self.update_image)
        self.threshold_scale_1.bind("<Motion>", self.update_image)

        # Initial display
        self.update_image()

    def update_image(self, event=None):
        # Get the threshold value from the scale
        threshold = self.threshold_scale.get()
        threshold_1 = self.threshold_scale_1.get()

        # thresh_img = ContourImg.identify_islands(self.original_img, threshold)
        thresh_img = ContourImg.detect_edges_rgb(self.original_img, threshold)
        # thresh_img = ContourImg.fill_contours(self.original_img, threshold, 0)
        # thresh_img = ContourImg.detect_edges_rgb(self.original_img, threshold)

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
    if (False):
        # Call main function with the desired number of main colors
        GetMainColorMap().detect('./texture_map.png', num_colors=5)

    else:
        root = tk.Tk()
        # app = ImageApp(root, './labeled_texture_map_1.png')
        app = ImageAppUsingML(root, texture_map, 7)
        root.mainloop()
