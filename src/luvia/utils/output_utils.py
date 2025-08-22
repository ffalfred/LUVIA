import cv2
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import uuid
 


class OutUtils:

    def __init__(self, base_folder):

        self.output_folder, self.name = self.create_outfolder(base_folder)
        (self.img_path, self.lineimg_path,
            self.character_img_path, self.cnnimg_path) = self.make_subfolders()

    def make_subfolders(self):
        img_path = Path(self.output_folder) / "images"
        # Create the directory
        img_path.mkdir(parents=True, exist_ok=True)
        charimg_path = Path(img_path) / "character_images"
        # Create the directory
        charimg_path.mkdir(parents=True, exist_ok=True)
        lineimg_path = Path(img_path) / "line_images"
        # Create the directory
        lineimg_path.mkdir(parents=True, exist_ok=True)
        cnnimg_path = Path(img_path) / "cnn_images"
        # Create the directory
        cnnimg_path.mkdir(parents=True, exist_ok=True)
        return img_path, lineimg_path, charimg_path, cnnimg_path

    def create_outfolder(self, base_folder):
        # Get current date and time
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Generate a 12-character unique ID using UUID
        unique_id = uuid.uuid4()
        name_folder = "LUVIA-RUN_{}_{}".format(timestamp,unique_id)
        # Build the path
        folder_path = Path(base_folder) / name_folder
        # Create the directory
        folder_path.mkdir(parents=True, exist_ok=True)
        # Return the absolute path
        return folder_path.resolve(), name_folder

    def save_projection_image(self, image_segments, projection, minima,
                                maxima, prefix, inverse=False):
        if inverse:
            # Save inverted
            image_segments = cv2.bitwise_not(image_segments)
        fig, (ax1, ax2) = plt.subplots(2,1)
        ax1.imshow(image_segments, cmap="gray")
        ax1.set_title("Vertical Projection Segmentation")
        ax1.set_xlabel("X-axis (pixels)")
        ax1.set_ylabel("Y-axis (pixels)")
        ax2.plot(projection, label="Smoothed Projection", color="blue")
        if len(minima) > 0:
            ax2.scatter(minima, projection[minima], color="red", label="Local Minima")
        if len(maxima) > 0:
            ax2.scatter(maxima, projection[maxima], color="green", label="Local Maxima")
        ax2.set_title("Vertical Projection Profile with Local Extrema")
        ax2.set_xlabel("Column Index")
        ax2.set_ylabel("Sum of Pixel Values")
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig("{}/{}.jpg".format(self.lineimg_path, prefix))
        plt.close()

    def save_image(self, image, prefix, folder="base", angle=0, scale=True,
                    inverse=True):
        if folder == "base":
            folder_save = self.img_path
        elif folder == "line":
            folder_save = self.lineimg_path
        elif folder == "character":
            folder_save = self.character_img_path
        # Save vanilla
        cv2.imwrite("{}/{}.jpg".format(folder_save, prefix), image)
        if scale:
            # Save vanilla + matrix
            plt.figure()
            plt.imshow(image, cmap="gray")
            plt.title("{}_{}".format(self.name, prefix))
            plt.xlabel("X-axis (pixels)")
            plt.ylabel("Y-axis (pixels)")
            plt.grid(True)
            plt.savefig("{}/{}_scale.jpg".format(folder_save, prefix))
            plt.close()
        if inverse:
            # Save inverted
            inverted_image = cv2.bitwise_not(image)
            cv2.imwrite("{}/{}_i.jpg".format(folder_save, prefix), inverted_image)
        if scale and inverse:
            # Save inverted + matrix
            plt.figure()
            plt.imshow(inverted_image, cmap="gray")
            plt.title("{}_{}_invert".format(self.name, prefix))
            plt.xlabel("X-axis (pixels)")
            plt.ylabel("Y-axis (pixels)")
            plt.grid(True)
            plt.savefig("{}/{}_i_scale.jpg".format(folder_save, prefix))
            plt.close()
    
    def plot_feature_maps(self, activation, prefix, num_maps=8):
        activation = activation.squeeze(0)
        fig, axes = plt.subplots(1, num_maps, figsize=(15, 5))
        for i in range(num_maps):
            axes[i].imshow(activation[i], cmap='viridis')
            axes[i].axis('off')
            axes[i].set_title("{}_{}".format(self.name, prefix))
        plt.tight_layout()
        plt.savefig("{}/{}.jpg".format(self.cnnimg_path, prefix))
        plt.close()

    # Maximally Activated Patches
    def maximally_activated_patches(self, activation, prefix, num_patches=8):
        activation = activation.squeeze(0)
        fig, axes = plt.subplots(1, num_patches, figsize=(15, 5))
        for i in range(num_patches):
            fmap = activation[i]
            axes[i].imshow(fmap, cmap='magma')
            axes[i].axis('off')
            axes[i].set_title("{}_{}".format(self.name, prefix))
        plt.tight_layout()
        plt.savefig("{}/{}.jpg".format(self.cnnimg_path, prefix))
        plt.close()

    # Filter Visualization
    def plot_filters(self, layer_weights, prefix, num_filters=8):
        weights = layer_weights.detach().cpu()
        fig, axes = plt.subplots(1, num_filters, figsize=(15, 5))
        for i in range(num_filters):
            axes[i].imshow(weights[i][0], cmap='gray')
            axes[i].axis('off')
            axes[i].set_title("{}_{}".format(self.name, prefix))
        plt.tight_layout()
        plt.savefig("{}/{}.jpg".format(self.cnnimg_path, prefix))
        plt.close()

    def plot_saliency(self, saliency, prefix):
        plt.figure(figsize=(6, 6))
        plt.imshow(saliency, cmap='hot')
        plt.axis('off')
        plt.title("{}_{}".format(self.name, prefix))
        plt.savefig("{}/{}.jpg".format(self.cnnimg_path, prefix))
        plt.close()

    def plot_sensitivity(self, sensitivity, prefix):
        plt.figure(figsize=(6, 6))
        plt.imshow(sensitivity, cmap='coolwarm')
        plt.axis('off')
        plt.title("{}_{}".format(self.name, prefix))
        plt.savefig("{}/{}.jpg".format(self.cnnimg_path, prefix))
        plt.close()

    def plot_guidedbackprop(self, gb_grad, prefix):
    
        plt.figure(figsize=(6, 6))
        plt.imshow(gb_grad, cmap='inferno')
        plt.axis('off')
        plt.title("{}_{}".format(self.name, prefix))
        plt.savefig("{}/{}.jpg".format(self.cnnimg_path, prefix))
        plt.close()


