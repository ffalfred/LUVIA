import cv2
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import uuid
import os
import string
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, Paragraph, PageTemplate, BaseDocTemplate, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, ListFlowable, ListItem
from reportlab.platypus import Image, PageBreak, Table, TableStyle, Paragraph
import os

from luvia.utils.pdf_utils import FormalReport


class OutUtils:

    def __init__(self, base_folder, mode):
        self.base_folder = base_folder

        self.output_folder, self.name = self.create_outfolder(base_folder, mode)
        if mode == "main":
            (self.img_path, self.lineimg_path,
                self.character_img_path, self.cnnimg_path) = self.make_subfolders()
        else:
            img_path = Path(self.output_folder) / "images"
            # Create the directory
            img_path.mkdir(parents=True, exist_ok=True)
        self.image_paths = {}

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

    def create_outfolder(self, base_folder, mode):
        if mode != "horde":
            # Get current date and time
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # Generate a 12-character unique ID using UUID
            unique_id = uuid.uuid4()
            name_folder = "LUVIA-RUN-{}_{}_{}".format(mode, timestamp,unique_id)
            # Build the path
            folder_path = Path(base_folder) / name_folder
            # Create the directory
            folder_path.mkdir(parents=True, exist_ok=True)
            # Return the absolute path
        else:
            name_folder = os.path.basename(base_folder)
            folder_path = Path(base_folder)
            folder_path.mkdir(parents=True, exist_ok=True)
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

    def save_image(self, image, prefix, suffix, folder="base", angle=0, scale=True,
                    inverse=True):
        if folder == "base":
            folder_save = self.img_path
        elif folder == "line":
            folder_save = self.lineimg_path
        elif folder == "character":
            folder_save = self.character_img_path
        if inverse:
            # Save inverted
            image = cv2.bitwise_not(image)
        img_path = "{}/{}_{}.jpg".format(folder_save, prefix, suffix)
        self.image_paths[suffix] = img_path
        if scale:
            # Save vanilla + matrix
            plt.figure()
            plt.imshow(image, cmap="gray")
            plt.title("{}_{}_{}".format(prefix, self.name, suffix))
            plt.xlabel("X-axis (pixels)")
            plt.ylabel("Y-axis (pixels)")
            plt.grid(True)
            plt.savefig(img_path)
            plt.close()
        else:
            # Save vanilla
            cv2.imwrite(img_path, image)
        
    
    def plot_feature_maps(self, activation, prefix, suffix, num_maps=9):

        activation = activation.squeeze(0)  # Shape: (8, H, W)
        fig, axes = plt.subplots(3, 3, figsize=(10, 12), facecolor='black')  # 4 rows, 2 columns
        # Ensure axes is a flat list for consistent indexing
        axes = axes.flatten()
        for i in range(num_maps):
            ax = axes[i]
            ax.tick_params(colors='white')
            ax.imshow(activation[i], cmap='viridis')
            ax.axis('off')
            axes[i].set_facecolor('black')
            ax.set_title(f"{self.name}_{prefix}", color='white')
        plt.tight_layout()
        path = f"{self.cnnimg_path}/{prefix}_{suffix}.jpg"
        self.image_paths[suffix+"_dict"][prefix] = path
        plt.subplots_adjust(hspace=0.0, wspace=0.05)
        plt.savefig(path)
        plt.close()


    # Maximally Activated Patches
    def maximally_activated_patches(self, activation, prefix, suffix, num_patches=9):
        activation = activation.squeeze(0)
        fig, axes = plt.subplots(3, 3, figsize=(10, 12), facecolor='black')  # 4 rows, 2 columns
        axes = axes.flatten()
        for i in range(num_patches):
            fmap = activation[i]
            axes[i].set_facecolor('black')
            axes[i].tick_params(colors='white')
            axes[i].imshow(fmap, cmap='magma')
            axes[i].axis('off')
            axes[i].set_title("{}_{}".format(self.name, prefix), color='white')
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.0, wspace=0.05)
        path = "{}/{}_{}.jpg".format(self.cnnimg_path, prefix, suffix)
        self.image_paths[suffix+"_dict"][prefix] = path
        plt.savefig(path)
        plt.close()

    # Filter Visualization
    def plot_filters(self, layer_weights, prefix, suffix, num_filters=9):
        weights = layer_weights.detach().cpu()
        fig, axes = plt.subplots(3, 3, figsize=(10, 12), facecolor='black')  # 4 rows, 2 columns
        axes = axes.flatten()
        for i in range(num_filters):
            axes[i].set_facecolor('black')
            axes[i].imshow(weights[i][0], cmap='gray')
            axes[i].axis('off')
            axes[i].tick_params(colors='white')
            axes[i].set_title("{}_{}".format(self.name, prefix), color='white')
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.0, wspace=0.05)
        path = "{}/{}_{}.jpg".format(self.cnnimg_path, prefix, suffix)
        self.image_paths[suffix+"_dict"][prefix] = path
        plt.savefig(path)
        plt.close()

    def plot_saliency(self, saliency, prefix, suffix):
        plt.figure(figsize=(6, 6), facecolor='black')
        plt.imshow(saliency, cmap='hot')

        plt.axis('on')
        plt.xticks(color='white')
        plt.yticks(color='white')
        plt.title("{}_{}".format(self.name, prefix), color='white')
        path = "{}/{}_{}.jpg".format(self.cnnimg_path, prefix, suffix)
        self.image_paths[suffix+"_dict"][prefix] = path
        plt.savefig(path)
        plt.close()

    def plot_sensitivity(self, sensitivity, prefix, suffix):
        plt.figure(figsize=(6, 6), facecolor='black')
        plt.imshow(sensitivity, cmap='coolwarm')
        plt.axis('on')
        plt.xticks(color='white')
        plt.yticks(color='white')
        plt.title("{}_{}".format(self.name, prefix), color='white')
        path = "{}/{}_{}.jpg".format(self.cnnimg_path, prefix, suffix)
        self.image_paths[suffix+"_dict"][prefix] = path
        plt.savefig(path)
        plt.close()

    def plot_guidedbackprop(self, gb_grad, prefix, suffix):
    
        plt.figure(figsize=(6, 6), facecolor='black')
        plt.imshow(gb_grad, cmap='inferno')
        plt.axis('on')
        plt.xticks(color='white')
        plt.yticks(color='white')
        plt.title("{}_{}".format(self.name, prefix), color='white')
        path = "{}/{}_{}.jpg".format(self.cnnimg_path, prefix, suffix)
        self.image_paths[suffix+"_dict"][prefix] = path
        plt.savefig(path)
        plt.close()

    def plot_alltransformations(self):
        fig, axes = plt.subplots(2, 2, figsize=(10, 12), facecolor='black')
        img = mpimg.imread(self.image_paths["original"])
        axes[0,0].imshow(img)
        axes[0,0].axis('off')        
        img = mpimg.imread(self.image_paths["cleaned"])
        axes[0,1].imshow(img)
        axes[0,1].axis('off')   
        img = mpimg.imread(self.image_paths["rotated"])
        axes[1,0].imshow(img)
        axes[1,0].axis('off')   
        img = mpimg.imread(self.image_paths["contours"])
        axes[1,1].imshow(img)
        axes[1,1].axis('off')   
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.tight_layout()
        path = "{}/images/image-transformation.jpg".format(self.base_folder)
        plt.savefig(path)
        plt.close()   
        
    def plot_allsentence_images(self, line_num, amount_charact):
        fig, axes = plt.subplots(nrows=9, ncols=amount_charact, facecolor="black",
                                        figsize=(amount_charact * 2, 18),
                                        squeeze=False)
        #axes = np.atleast_2d(axes)
        print("HELLO", axes.shape, amount_charact)

        for s in range(amount_charact):
            name_key = "line-{}_character-{}_dict".format(line_num, s)
            images = self.image_paths[name_key]
            img = mpimg.imread(self.image_paths["image_line-{}_character-{}".format(line_num, s)])
            axes[0,s].imshow(img)
            axes[0,s].axis('off')
            img = mpimg.imread(images["cnn_featmap2"])
            axes[4,s].imshow(img)
            axes[4,s].axis('off')
            img = mpimg.imread(images["cnn_featmap1"])
            axes[5,s].imshow(img)
            axes[5,s].axis('off')
            img = mpimg.imread(images["cnn_actMAX1"])
            axes[6,s].imshow(img)
            axes[6,s].axis('off')
            img = mpimg.imread(images["cnn_saliency"])
            axes[1,s].imshow(img)
            axes[1,s].axis('off')
            img = mpimg.imread(images["cnn_guidedbackprop"])
            axes[2,s].imshow(img)
            axes[2,s].axis('off')
            img = mpimg.imread(images["cnn_sensitivity"])
            axes[3,s].imshow(img)
            axes[3,s].axis('off')
            img = mpimg.imread(images["cnn_act1"])
            axes[7,s].imshow(img)
            axes[7,s].axis('off')
            img = mpimg.imread(images["cnn_act2"])
            axes[8,s].imshow(img)
            axes[8,s].axis('off')        
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.tight_layout()
        path = "{}/sentence-spectrum.jpg".format(self.cnnimg_path)
        plt.savefig(path)
        plt.close()                



    def plot_allchar_images(self, suffix):
        images = self.image_paths[suffix+"_dict"]
        fig, axes = plt.subplots(3, 3, figsize=(10, 12), facecolor='black')
        axes = axes.ravel()
        img = mpimg.imread(self.image_paths["image_"+suffix])
        axes[0].imshow(img)
        axes[0].axis('off')
        img = mpimg.imread(images["cnn_featmap2"])
        axes[4].imshow(img)
        axes[4].axis('off')
        img = mpimg.imread(images["cnn_featmap1"])
        axes[5].imshow(img)
        axes[5].axis('off')
        img = mpimg.imread(images["cnn_actMAX1"])
        axes[6].imshow(img)
        axes[6].axis('off')
        img = mpimg.imread(images["cnn_saliency"])
        axes[1].imshow(img)
        axes[1].axis('off')
        img = mpimg.imread(images["cnn_guidedbackprop"])
        axes[2].imshow(img)
        axes[2].axis('off')
        img = mpimg.imread(images["cnn_sensitivity"])
        axes[3].imshow(img)
        axes[3].axis('off')
        img = mpimg.imread(images["cnn_act1"])
        axes[7].imshow(img)
        axes[7].axis('off')
        img = mpimg.imread(images["cnn_act2"])
        axes[8].imshow(img)
        axes[8].axis('off')
        for ax in axes:
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
                spine.set_linewidth(2)

        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.tight_layout()
        path = "{}/character-spectrum_{}.jpg".format(self.cnnimg_path, suffix)
        plt.savefig(path)
        plt.close()

    def create_pdfimage(self):
        report = FormalReport("{}/LUVIA_reportimage.pdf".format(self.output_folder))
        report.add_cover_page(project_name="LUVIA Analysis - Image Scrapping", author="Alfred Ferrer Florensa", date="27/08/2025")
        # Section with image
        report.add_section_with_image(title="Original traces found on the asphalt",
                                    text="This is the uploaded image of the traces found on the asphalt",
                                    image_path=self.image_paths["original"])
        # Section with image
        report.add_section_with_image(title="Smedt shorthand detected on the asphalt",
                                    text="LUVIA found traces on the street with high chances of being Smedt shorthand",
                                    image_path=self.image_paths["cleaned"])
        # Section with image
        report.add_section_with_image(title="Oriented traces of the Smedt shorthand",
                                    text="LUVIA has oriented the street into the direction of the traces of the Smedt shorthand",
                                    image_path=self.image_paths["rotated"])
        # Section with image
        report.add_section_with_image(title="Found Smedt shorthand sentences",
                                    text="LUVIA has detected sentences of Smedt shorthand on the street",
                                    image_path=self.image_paths["contours"])
        report.build()


    def create_pdftranslation(self, sentences_data):

        report = FormalReport("{}/LUVIA_reporttranslation.pdf".format(self.output_folder))
        report.add_cover_page(project_name="LUVIA Analysis - Translation", author="Alfred Ferrer Florensa", date="27/08/2025")

        # Section with image
        report.add_section_with_image(title="Sentences found by LUVIA written in Smedt shorthand",
                                    text="Luvia detected {} possible sentences hidden in the asphalt".format(len(sentences_data)),
                                    image_path=self.image_paths["contours"])
        report.story.append(PageBreak())
        report.add_section(title="Individual analysis of sentences detected", content="Below you can find a extensive analysis of the written words found by LUVIA")
        for idx, entry in enumerate(sentences_data):
            report.add_subsection_with_image(title="Sentence number {}".format(idx), location="52,60"
                                             ,proposed_sentences=entry, image_path=self.image_paths["image_line-{}".format(idx)])
            report.story.append(PageBreak())

        report.build()

if __name__== "__main__":

    from reportlab.lib.units import inch

    report = FormalReport("my_report_with_image.pdf")
    report.add_cover_page(project_name="LUVIA Analysis", author="Alfred Ferrer Florensa", date="27/08/2025")

    # Section with image

    report.add_section_with_image("Sentences found by LUVIA written in Smedt shorthand", "Luvia detected 5 possible sentences hidden in the asphalt",
                                  "test/LUVIA-RUN_2025-08-26_00-09-37_03c19400-9a8e-4ca2-9c63-07d42d588f5f/images/contours_i_scale.jpg")
    report.add_section("Individual analysis of sentences detected", "Below you can find a extensive analysis of the written words found by LUVIA")
    report.build()



