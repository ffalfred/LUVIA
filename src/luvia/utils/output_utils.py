import cv2
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import uuid
import os
import string
import numpy as np
import matplotlib.gridspec as gridspec

from PIL import ImageFilter
from PIL import Image as IMAGEPIL
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, Paragraph, PageTemplate, BaseDocTemplate, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, ListFlowable, ListItem
from reportlab.platypus import Image, PageBreak, Table, TableStyle, Paragraph
import os
from luvia.utils.image_utils import ImageUtils
import textwrap

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
        self.image_objects = {"general": {}, "lines": {}}

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
                                maxima, prefix, line_count, inverse=False):
        if inverse:
            # Save inverted
            image_segments = cv2.bitwise_not(image_segments)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,8),
                                       gridspec_kw={'height_ratios': [1.25, 1]})

        # Top image plot
        ax1.imshow(image_segments, cmap="gray")
        ax1.set_title("Vertical Projection Segmentation")
        ax1.set_xlabel("X-axis (pixels)")
        ax1.set_ylabel("Y-axis (pixels)")
        ax1.tick_params()
        ax1.grid(True)

        # Bottom projection plot
        ax2.plot(projection, label="Smoothed Projection", color="blue")
        if len(minima) > 0:
            minima_flatten = [item for tup in minima for item in tup]
            ax2.scatter(minima_flatten, [int(projection[i]) for i in minima_flatten], color="red", label="Local Minima")
        if len(maxima) > 0:
            #maxima_flatten = [item for tup in maxima for item in tup]
            ax2.scatter(maxima, [int(projection[i]) for i in maxima], color="green", label="Local Maxima")
        ax2.set_title("Vertical Projection Profile with Local Extrema")
        ax2.set_xlabel("Column Index")
        ax2.set_ylabel("Sum of Pixel Values")
        ax2.legend()
        ax2.grid(True)
        ax2.tick_params()
        path = f"{self.lineimg_path}/{prefix}.jpg"
        self.image_paths["line-{}-projection".format(line_count)] = path
        # Final layout and save
        plt.tight_layout()
        plt.savefig(path, facecolor=fig.get_facecolor(), dpi=300)
        plt.close()


    def save_image(self, image, prefix, suffix, folder="base", angle=0, scale=True,
                    inverse=True, general=False):
        if folder == "base":
            folder_save = self.img_path
        elif folder == "line":
            folder_save = self.lineimg_path
        elif folder == "character":
            folder_save = self.character_img_path
        if inverse:
            # Save inverted
            image = ImageUtils.invert_image(image)
            if general:
                image = IMAGEPIL.fromarray(np.uint8(image)).convert("L")
                thickened = image.filter(ImageFilter.MaxFilter(size=3))
                image = np.array(thickened)
        image = ImageUtils.rotate_image(image=image, angle=angle)
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
            plt.savefig(img_path, dpi=300)
            plt.close()
        else:
            # Save vanilla
            cv2.imwrite(img_path, image, dpi=300)
        
    
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
        plt.savefig(path, dpi=300)
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
        plt.savefig(path, dpi=300)
        plt.close()

    def plot_saliency(self, saliency, prefix, suffix):
        plt.figure(figsize=(6, 6), facecolor='black')
        img = plt.imshow(saliency, cmap='hot')

        plt.axis('on')
        plt.xticks(color='white')
        plt.yticks(color='white')
        plt.title("{}_{}".format(self.name, prefix), color='white')
        path = "{}/{}_{}.jpg".format(self.cnnimg_path, prefix, suffix)
        self.image_paths[suffix+"_dict"][prefix] = path
        plt.savefig(path, dpi=300)
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
        plt.savefig(path, dpi=300)
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
        plt.savefig(path, dpi=300)
        plt.close()

    def plot_alltransformations(self):
        fig = plt.figure(figsize=(10, 12), facecolor='black')
        plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03, hspace=0.01, wspace=0.03)
        gs = gridspec.GridSpec(2, 2, height_ratios=[1.5, 3], hspace=0.01, wspace=0.01)
        # Subplots
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])

        img = mpimg.imread(self.image_paths["original"])
        ax1.imshow(img)
        ax1.axis('off')        
        img = mpimg.imread(self.image_paths["cleaned"])
        ax2.imshow(img)
        ax2.axis('off')   
        img = mpimg.imread(self.image_paths["contours"])
        ax3.imshow(img)
        ax3.axis('off')   
        #plt.tight_layout()
        path = "{}/image-transformation.jpg".format(self.output_folder)
        plt.savefig(path, dpi=300)
        plt.close()   
        
         
    def plot_allchar_images(self, suffix, line_count, word, sentence_info):
        images = self.image_paths[suffix+"_dict"]
        fig = plt.figure(figsize=(10, 14), facecolor='black')
        sentence = sentence_info["sentence"]
        probability = sentence_info["probability"]
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, hspace=0.05, wspace=0.5)
        #plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.05, hspace=0.05, wspace=0.05)

        fig.patch.set_facecolor('black')  # Ensure full black background
        # Add an extra row for the title (5 rows total)
        gs = gridspec.GridSpec(4, 3, height_ratios=[3.5, 0.3, 2, 2], hspace=0.1, wspace=0.05)
        # Subplots
        ax1 = fig.add_subplot(gs[2, 0])
        ax2 = fig.add_subplot(gs[2, 1])
        ax3 = fig.add_subplot(gs[2, 2])
        ax4 = fig.add_subplot(gs[3, 0])
        ax5 = fig.add_subplot(gs[3, 1])
        ax6 = fig.add_subplot(gs[3, 2])
        title_ax = fig.add_subplot(gs[1, :])  # Title axis
        ax0 = fig.add_subplot(gs[0, :])
        axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax0]
        img = mpimg.imread(self.image_paths["image_"+suffix])
        ax1.imshow(img)
        ax1.set_facecolor('black')
        ax1.axis('off')
        img = mpimg.imread(images["cnn_saliency"])
        ax2.imshow(img)
        ax2.set_facecolor('black')
        ax2.axis('off')
        img = mpimg.imread(images["cnn_guidedbackprop"])
        ax3.imshow(img)
        ax3.set_facecolor('black')
        ax3.axis('off')
        img = mpimg.imread(images["cnn_actMAX1"])
        ax5.imshow(img)
        ax5.set_facecolor('black')
        ax5.axis('off')
        img = mpimg.imread(images["cnn_act1"])
        ax4.imshow(img)
        ax4.set_facecolor('black')
        ax4.axis('off')
        img = mpimg.imread(images["cnn_sensitivity"])
        ax6.imshow(img)
        ax6.set_facecolor('black')
        ax6.axis('off')
        img = mpimg.imread(self.image_paths["line-{}-projection".format(line_count)])
        ax0.imshow(img)
        ax0.set_facecolor('black')
        ax0.axis('off')        
        ax1_left = ax1.get_position().x0
        # Mid-level title axis
        title_ax.axis('off')
        title_ax.set_facecolor('black')

        title_ax.text(0.0, 0.5,
                    "Word Spectrum", ha='left', va='center',
                    fontsize=22, color='white', transform=title_ax.transAxes)

        fig.text(ax1_left, 0.975,
                "Sentence Fragmentation:", ha='left', va='top',
                fontsize=22, color='white')
        
        word_cleaned = word.translate(str.maketrans('', '', string.punctuation))
        # Subtitle for the word title
        title_ax.text(0.0, -0.5,  # slightly lower than 0.5
                    r"Word at Position {}: $\bf{{{}}}$".format(int(suffix[-1])+1, word_cleaned),
                    ha='left', va='center', fontsize=16, color='white',
                    transform=title_ax.transAxes)

        # Wrap the sentence
        wrapped_lines = textwrap.wrap(sentence, width=80)
        formatted_sentence = "   "
        formatted_sentence += "\n".join(wrapped_lines)

        fig.text(ax1_left, 0.945, "Sentence: ", ha='left', va='top', fontsize=16, color='white')
        # Display both
        fig.text(ax1_left+0.1, 0.945, formatted_sentence, ha='left', va='top', fontsize=16, color='white', fontweight='bold')

        for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
                spine.set_linewidth(2)
        fig.patch.set_facecolor('black') 
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.tight_layout()
        path = "{}/cs_{}.jpg".format(self.output_folder, suffix.replace("character", "char"))
        plt.savefig(path, dpi=300, facecolor=fig.get_facecolor())
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


    def create_pdftranslation(self, user, character, location, sentences_data):

        report = FormalReport("{}/LUVIA_reporttranslation.pdf".format(self.output_folder),
                              location=location, agent=character)
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d %H:%M:%S")

        report.add_cover_page(project_name="LUVIA Analysis - The Non-Quasisplit Case",
                              author=user, date=date_time)

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



